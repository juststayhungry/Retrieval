import torch
import clip
import numpy as np
from PIL import Image
from tqdm import *
import os
from utils import set_seed, setup_logger, load_config_file
from data import sen_parse

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 定义函数，对输入的image进行编码
def encode_image(image):
    image = preprocess(Image.open(image)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features.cpu()

# 定义函数，返回和文本查询的相似性得分
@torch.no_grad()
def get_similarities(text_features, image_features, batch_size=128):
    # text = clip.tokenize([query]).to(device)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    num_rows = image_features.shape[0]
    all_sims = []
    for i in tqdm(range(0, num_rows, batch_size)):
        end = min(i + batch_size, num_rows)
        batch_features = image_features[i:end, :]
        
        similarities = (100.0 * text_features @ batch_features.T).softmax(dim=-1)#squeeze(0)
        all_sims.append(similarities)
    all_sims = torch.cat(all_sims).cpu()
    
    return all_sims


def evaluate(data_file_path,unseen_comp_file_path, k):
  
    img_id_list = []
    caption_list = []
    obj_list =[]
    #读取image_id对应的图像以及其caption
    with open(unseen_comp_file_path, "r") as file:
          # 跳过第一行
        next(file)
        for line in file:
            # 去除行末尾的换行符
          line = line.strip()
          if line and line.count('\t') >= 1:#排除空行
            # 使用空格分割每一行的内容
            parts = line.split("\t")
            # 获取 ID 和 Caption
            image_id = parts[0]
            caption = " ".join(parts[1:])

            img_id_list.append(int(image_id))
            caption_list.append(caption)
            obj,_,_= sen_parse(caption=caption)
            obj_list.append(obj)
            # caption2imgid = {str(caption):img_id for img_id ,caption in zip(img_id_list,caption_list)}
            # imgid2caption = {img_id:caption for img_id ,caption in zip(img_id_list,caption_list)}

    # 初始化计数器
    total = len(img_id_list)
    correct = 0
    image_features_list = []
    text_all = torch.cat([clip.tokenize(f"{c}" for c in caption_list)]).to(device)#，直接将caption作为CLIP_text的输入
    # text_all = torch.cat([clip.tokenize(f"{c}" for c in obj_list)]).to(device)#将每张图像中所有的object作为CLIP_text的输入
    with torch.no_grad():
        text_features = model.encode_text(text_all).to("cpu")
    
    # 对于每个图像...
    for image_id in tqdm(img_id_list):
        # 获取图像的文件路径和标注
        filename = "000000" + str(image_id).zfill(6) + ".jpg"
        image_path = os.path.join(data_file_path,filename)
        with torch.no_grad():
            image_features = encode_image(image_path)
        image_features_list.append(image_features)

    image_features = torch.cat(image_features_list)

    similarities = get_similarities(text_features, image_features)
    for i in range(len(img_id_list)):
        values,indices = similarities[i].topk(k)
        if i in indices:
          correct += 1
    # 计算recall@k并返回
    recall_at_k = float(correct) / float(total)
    return recall_at_k



if __name__ == '__main__':
    
    DATA_CONFIG_PATH = 'data/data_config.yaml'
    data_config = load_config_file(DATA_CONFIG_PATH)
    # MS COCO图像   以及    unseen_comp{img_id,caption}的path
    data_file_path = data_config.train_img_dir
    unseen_comp_file_path = data_config.unseen_imgid_caption_dir
    

    k = 1#recall@k
    recall_at_5 = evaluate(data_file_path,unseen_comp_file_path, k)
    print("Recall@{}: {:.4f}".format(k,recall_at_5))

