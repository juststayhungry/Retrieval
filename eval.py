import torch
import os
import clip
import numpy as np
from PIL import Image
import ipdb
from tqdm import *
from utils import mkdir, setup_logger, load_config_file
from data.datasets import CLIP_COCO_dataset,CompositionDataset
from data.data_loaders import DATASET_PATHS
from model.modules import get_model
from omegaconf import OmegaConf
from utils.simple_tokenizer import SimpleTokenizer
import argparse


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
        similarities = (100.0 * batch_features@text_features.T).softmax(dim=-1)
        all_sims.append(similarities)
    all_sims = torch.cat(all_sims).cpu()
    
    return all_sims

def save_data(file_path,data):
    text = 'recall @1   @5  @10：\n'
    for i in data:
        text += str(i)+'\t' 
    with open(file_path, 'w') as f:
        f.write(text)

def evaluate(model,config,eval_loader,k_list):
    logger.info("***** Running training *****")
    model.eval()
    image_features_list = []
    text_features_list = []
    pairs_id_list = []
    recall_i2t = []
    recall_t2i = []
    with torch.no_grad():
      for i, batch in tqdm(enumerate(eval_loader)):
          if config.dataset == "coco":
            image ,caption = batch #B D
          else:
            image ,caption,pairs_id = batch #B D
          image = image.to(config.device)#???
          text = caption.to(config.device)
          with torch.no_grad():
              text_feature = model.encode_text(text)
              image_feature = model.encode_image(image) 
          text_features_list.append(text_feature.to("cpu"))
          image_features_list.append(image_feature.to("cpu"))
          if config.dataset == "coco":
              pairs_id_list = [i for i in range(total)]
          else:
              pairs_id_list.extend(pairs_id.to("cpu"))
      text_features = torch.cat(text_features_list, dim=0)
      image_features = torch.cat(image_features_list, dim=0)
      similarities_i2t = get_similarities(image_features.float(), text_features.float(),batch_size=config.batch_size)
    similarities_t2i = similarities_i2t.T
    for k in k_list:
      correct_i2t =0
      correct_t2i =0
      for i in range(total):#遍历每张图像，检索k个文本id
          values,indices = similarities_i2t[i].topk(k)#
          pairs_id_list1 = [pairs_id_list[j] for j in indices]
          if pairs_id_list[i] in pairs_id_list1:
            correct_i2t += 1
          values,indices = similarities_t2i[i].topk(k)##遍历每个文本，检索k个图像id         
          if i in indices:
            correct_t2i += 1
    # 计算recall@k并返回
      recall_i2t.append(float(correct_i2t) / float(total))
      recall_t2i.append(float(correct_t2i) / float(total))
    
    return recall_i2t,recall_t2i



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",default= 'cgqa',type=str, required=False, help="path of the data for evaluating")
    parser.add_argument("--openai_clip_model",default="ViT-B/32",type=str,choices=["RN101","RN50","ViT-B/32"],required=False,help="select model type")
    parser.add_argument("--batch-size",default=16,type=int,required=False,help="evaluate batchsize")
    parser.add_argument("--pretrained-pt",required=True,type=str,help="path/to/checkpoints/epoch_K.pt")
    parser.add_argument("--save-dir",type=str,required=False,help="dir/to/eval/result")
    args = parser.parse_args()

    EVAL_CONFIG_PATH = 'eval_config.yaml'
    config = load_config_file(EVAL_CONFIG_PATH)
    
    config.train_type = 'eval'
    if args.openai_clip_model:#open ai pretrained model's name
        config.MODEL.BACKBONE.NAME =  args.openai_clip_model
    if args.dataset:
        config.dataset = args.dataset
    if args.pretrained_pt:
        config.pretrained_pt = args.pretrained_pt
    if args.save_dir:
        config.save_dir = args.save_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.img_dir = DATASET_PATHS[config.dataset]
    
    eval_result_dir =  os.path.join(config.save_dir, config.dataset)
    mkdir(path=eval_result_dir)
    
    tokenizer = SimpleTokenizer()
    if config.dataset == "coco":
        eval_dataset = CLIP_COCO_dataset(config,tokenizer)
    else:
        eval_dataset = CompositionDataset(config,tokenizer)
    total = len(eval_dataset)
    global logger
    logger = setup_logger('Eval on {}'.format(config.dataset), eval_result_dir, 0, filename = "training_logs.txt")
    logger.info("there are {} eval_datas in total".format(total))
    eval_loader = torch.utils.data.DataLoader(eval_dataset,batch_size=config.batch_size,shuffle=False)#
    logger.info(f"evaluation parameters {config}")
    logger.info("dataset : '{}'".format(config.dataset))
    
    #load OPEN AI pretrained CLIP
    openai_pretrained_model, _ = clip.load(config.MODEL.BACKBONE.NAME, device=config.device)

    #2.load our trained CLIP
    model= get_model(config)
    # loading trained weights
    checkpoint = torch.load(config.pretrained_pt)#load from trained path
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model = model.to(config.device)
    
    k_list = [1,5,10]#recall@k
    
#     recall1,_ = evaluate(openai_pretrained_model,config,eval_loader,k_list)
#     result_path1 =  os.path.join(eval_result_dir,f'b1_result.txt')
#     logger.info("Openai_pretrained_clip {} recall @1   @5  @10: \n{}  {}  {}" .format(config.MODEL.BACKBONE.NAME,recall1[0],recall1[1],recall1[2]))
#     save_data(result_path1,recall1)
    
    recall2,_ = evaluate(model,config,eval_loader,k_list)
    logger.info("Our trained_clip {} recall @1   @5  @10:\n{}  {}  {}" .format(config.pretrained_pt,recall2[0],recall2[1],recall2[2])) 
    result_path2 =  os.path.join(eval_result_dir,f'b22_result.txt')
    save_data(result_path2,recall2)
    
    logger.info("Evaling done")
    
    

