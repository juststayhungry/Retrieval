import torch
import clip
import numpy as np
from pycocotools.coco import COCO
from tqdm import *
from data.datasets import CustomDataset

# 定义函数，返回和文本查询的相似性得分
def i2t(images, captions, npts=None):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts == None:
        npts = images.size()[0] // 5

    ranks = np.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].unsqueeze(0)

        # Compute scores
        d = torch.mm(im, captions.t())
        d_sorted, inds = torch.sort(d, descending=True)
        inds = inds.data.squeeze(0).cpu().numpy()

        # Score
        rank = 1e20
        # find the highest ranking
        for i in range(5*index, 5*index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    return (r1, r5, r10, medr)

def t2i(images, captions, npts=None, data='f8k'):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts == None:
        npts = images.size()[0] // 5

    ims = torch.cat([images[i].unsqueeze(0) for i in range(0, len(images), 5)])

    ranks = np.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5*index : 5*index + 5]

        # Compute scores
        d = torch.mm(queries, ims.t())
        for i in range(d.size()[0]):
            d_sorted, inds = torch.sort(d[i], descending=True)
            inds = inds.data.squeeze(0).cpu().numpy()
            ranks[5 * index + i] = np.where(inds == index)[0][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    return (r1, r5, r10, medr)

def evaluate(data_file_path,unseen_comp_file_path, k):
  eval_dataset = CustomDataset(unseen_comp_file_path,data_file_path,preprocess)
  total = len(eval_dataset)
  batch_size = 100
  eval_loader = torch.utils.data.DataLoader(eval_dataset,batch_size=batch_size,drop_last=True)#
  image_features_list = []
  text_features_list = []
  for i, batch in tqdm(enumerate(eval_loader)):
    image ,caption = batch #B D
    text = torch.cat([clip.tokenize(f"{c}" for c in caption)]).to(device)
    with torch.no_grad():
      text_feature = model.encode_text(text)
      image_feature = model.encode_image(image) #B K
    #先分批次读取后再对特征进行汇总
    text_features_list.append(text_feature)
    text_features = torch.cat(text_features_list, dim=0)
    image_features_list.append(image_feature)
    image_features = torch.cat(image_features_list, dim=0)
 
  # return t2i(images=image_features,captions=text_features)
  return i2t(images=image_features,captions=text_features)


if __name__ == '__main__':
  # 加载MS COCO数据集和注释
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, preprocess = clip.load("ViT-B/32", device=device)
  data_file_path = '/content/train2017'
  # unseen_comp_file_path = "/content/gdrive/MyDrive/data_split/unseen_comp.txt"
  unseen_comp_file_path = "/content/test.txt"
  k = 1
  recall = evaluate(data_file_path,unseen_comp_file_path, k)
  print("r1, r5, r10, medr结果分别是",recall)
    
        