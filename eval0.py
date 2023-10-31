import torch
import clip
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
from tqdm import *
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 定义函数，对输入的image进行编码
def encode_image(image):
    image = preprocess(Image.open(image)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features.cpu()

# 定义函数，返回和文本查询的相似性得分
def get_similarities(text_features, image_features):
    # text = clip.tokenize([query]).to(device)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarities = (100.0 * text_features @ image_features.T).softmax(dim=-1)#squeeze(0)

    return similarities.cpu()


def evaluate(data_file_path,unseen_comp_file_path, k):
  
    img_id_list = []
    caption_list = []
    # 获取所有图像id
    # image_ids = coco.getImgIds()
    #读取image_id对应的图像以及其caption，直接将caption作为CLIP_text的输入
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
            # caption2imgid = {str(caption):img_id for img_id ,caption in zip(img_id_list,caption_list)}
            # imgid2caption = {img_id:caption for img_id ,caption in zip(img_id_list,caption_list)}

    # 初始化计数器
    total = len(img_id_list)
    correct = 0
    image_features_list = []
    text_all = torch.cat([clip.tokenize(f"{c}" for c in caption_list)]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_all)
    
    # 对于每个图像...
    for image_id in tqdm(img_id_list):
        
        # 获取图像的文件路径和标注
        filename = "000000" + str(image_id).zfill(6) + ".jpg"
        image_path = os.path.join(data_file_path,filename)
        with torch.no_grad():
            image_features = encode_image(image_path)
        image_features_list.append(image_features)

    image_features = torch.cat(image_features_list)
    print(image_features.size())

    similarities = get_similarities(text_features, image_features)
    for i in range(len(img_id_list)):
        values,indices = similarities[i].topk(k)
        if i in indices:
          correct += 1
    # 计算recall@k并返回
    recall_at_k = float(correct) / float(total)
    return recall_at_k



if __name__ == '__main__':
    # 加载MS COCO数据集和注释
    data_file_path = '/content/train2017'
    # unseen_comp_file_path = "/content/gdrive/MyDrive/data_split/unseen_comp.txt"
    unseen_comp_file_path = "/content/test.txt"
    k = 1
    recall_at_5 = evaluate(data_file_path,unseen_comp_file_path, k)
    print("Recall@{}: {:.4f}".format(k,recall_at_5))

