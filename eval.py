import torch
import clip
import numpy as np
from pycocotools.coco import COCO
from tqdm import *
from data.datasets import CustomDataset

# 定义函数，返回和文本查询的相似性得分
def get_similarities(text_features, image_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarities = (100.0 * text_features @ image_features.T).softmax(dim=-1)#squeeze(0)
    return similarities.cpu()

def evaluate(data_file_path,unseen_comp_file_path, k):
  eval_dataset = CustomDataset(unseen_comp_file_path,data_file_path,preprocess)
  total = len(eval_dataset)
  batch_size = 5
  eval_loader = torch.utils.data.DataLoader(eval_dataset,batch_size=batch_size,drop_last=True)#
  correct = 0
  for i, batch in tqdm(enumerate(eval_loader)):
  # for batch in tqdm(eval_dataset):?有区别？
    image ,caption = batch #B D
    text = torch.cat([clip.tokenize(f"{c}" for c in caption)]).to(device)
    with torch.no_grad():
      text_features = model.encode_text(text)
      image_features = model.encode_image(image) #B K   
    similarities = get_similarities(text_features, image_features)#候选的text应该是全集
    for j in range(batch_size):
      values,indices = similarities[j].topk(k)
      if j in indices:
        correct += 1
  # 计算recall@k并返回
  recall = float(correct) / float(total)
  return recall


if __name__ == '__main__':
  # 加载MS COCO数据集和注释
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, preprocess = clip.load("ViT-B/32", device=device)
  data_file_path = '/content/train2017'
  # unseen_comp_file_path = "/content/gdrive/MyDrive/data_split/unseen_comp.txt"
  unseen_comp_file_path = "/content/test.txt"
  k = 1
  recall_at_5 = evaluate(data_file_path,unseen_comp_file_path, k)
  print("Recall@{}: {:.4f}".format(k,recall_at_5))
    
        