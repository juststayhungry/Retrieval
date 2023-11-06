import torch
import os
import clip
import numpy as np
from PIL import Image
from tqdm import *
from utils import mkdir, setup_logger, load_config_file
from data.datasets import CLIP_COCO_dataset
from model.model import CLIP
from utils.simple_tokenizer import SimpleTokenizer


# 定义函数，返回和文本查询的相似性得分
@torch.no_grad()
def get_similarities(image_features,text_features,batch_size=128):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    num_rows = image_features.shape[0]
    all_sims = []
    for i in tqdm(range(0, num_rows, batch_size)):
        end = min(i + batch_size, num_rows)
        batch_features = image_features[i:end, :]
        #BK KN   BN i2t  
        similarities = (100.0 * batch_features@text_features.T).softmax(dim=-1)#squeeze(0)
        all_sims.append(similarities)
    all_sims = torch.cat(all_sims).cpu()
    
    return all_sims

def save_data(file_path,data):
    text = 'recall @1   @5  @10分别为：\n'
    for i in data:
        text += str(i)+'\t' 
    with open(file_path, 'w') as f:
        f.write(text)

def evaluate(model,data_config,k_list):
    tokenizer = SimpleTokenizer()
    eval_dataset = CLIP_COCO_dataset("eval",data_config,tokenizer)
    batch_size = 128
    total = len(eval_dataset)
    print("总共有{}个caption".format(total))
    eval_loader = torch.utils.data.DataLoader(eval_dataset,batch_size=batch_size)#
    image_features_list = []
    text_features_list = []
    recall_i2t = []
    recall_t2i = []
    for i, batch in tqdm(enumerate(eval_loader)):
        image ,caption = batch #B D
        image = image.to(device)#???
        text = caption.to(device)
        with torch.no_grad():
          text_feature = model.encode_text(text)
          image_feature = model.encode_image(image) 
        #先分批次读取后再对特征进行汇总
        text_features_list.append(text_feature.to("cpu"))
        image_features_list.append(image_feature.to("cpu"))
    text_features = torch.cat(text_features_list, dim=0)
    image_features = torch.cat(image_features_list, dim=0)
    similarities_i2t = get_similarities(image_features.float(), text_features.float(),batch_size)
    similarities_t2i = similarities_i2t.T
    for k in k_list:
      correct_i2t =0
      correct_t2i =0
      for i in range(total):#遍历每张图像，检索k个文本id
          values,indices = similarities_i2t[i].topk(k)#
          if i in indices:
            correct_i2t += 1
          values,indices = similarities_t2i[i].topk(k)##遍历每个文本，检索k个图像id         
          if i in indices:
            correct_t2i += 1
    # 计算recall@k并返回
      recall_i2t.append(float(correct_i2t) / float(total))
      recall_t2i.append(float(correct_t2i) / float(total))

    return recall_i2t,recall_t2i


if __name__ == '__main__':
    DATA_CONFIG_PATH = 'data/data_config.yaml'
    TRAINER_CONFIG_PATH = 'trainer/train_config.yaml'
    MODEL_CONFIG_PATH = 'model/model_config.yaml'

    data_config = load_config_file(DATA_CONFIG_PATH)
    train_config = load_config_file(TRAINER_CONFIG_PATH)
    mkdir(path=data_config.eval_result_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #1.eval on OPEN AI pretrained CLIP
    openai_pretrained_model, _ = clip.load("ViT-B/32", device=device)

    #2.eval on our trained CLIP
    model_config = load_config_file(MODEL_CONFIG_PATH)
    # creating  CLIP model
    model_params = dict(model_config.ViTB32)
    model = CLIP(**model_params)

    # loading trained weights
    model_path =  os.path.join(train_config.checkpoint_dir, f'checkpoint_36_50000.pt')#add checkpoint's name
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    k_list = [1,5,10]#recall@k
    recall1,_ = evaluate(openai_pretrained_model,data_config, k_list)
    result_path1 =  os.path.join(data_config.eval_result_dir, f'baseline1_result.txt')
    save_data(result_path1,recall1)
    
    recall2,_ = evaluate(model,data_config, k_list)
    result_path2 =  os.path.join(data_config.eval_result_dir, f'baseline2_result.txt')
    save_data(result_path2,recall2)
    

