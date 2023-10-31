import random
import os
import json
import sng_parser
import spacy
from tqdm import *

# 加载英文语言模型
nlp = spacy.load('en_core_web_sm')
def extract_adjectives(sentence):
    # 对句子进行分析
    doc = nlp(sentence)
    
    # 提取形容词
    adjectives = []
    for token in doc:
        if token.pos_ == 'ADJ':  # 判断词性是否为形容词
            adjectives.append(token.text)
    return adjectives

def sen_parse(caption): 
    #句子解析并输出三类单词；
    #输入caption；
    #输出三类单词的set
  attributes = set()
  relations = set()
  objects = set()
  graph = sng_parser.parse(caption)
  for i in range(len(graph.get("entities"))):
    objects.update({graph.get("entities")[i].get("head")})#对象

  for i in range(len(graph.get('relations'))):
    relations.update({graph.get('relations')[i].get("relation")})#关系
    
  attribute = extract_adjectives(caption)
  attributes.update(attribute)#属性
  return objects,attributes,relations

def split_dataset(dataset, split_ratio):
    random.shuffle(dataset)  # 随机打乱数据集顺序
    split_index = int(len(dataset) * split_ratio)
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]
    return train_data, test_data

captions_train_path = r"/content/annotations/captions_train2017.json"
# 读取json文件
with open(captions_train_path, 'r') as f1:
    dictortary = json.load(f1)

# 得到images和annotations信息
images_value = dictortary.get("images")
annotations_value = dictortary.get("annotations")
#对训练caption进行解析
attributes_train = set()
relations_train = set()
objects_train = set()

id_caption = dict()

#每个测试图像仅保留一个caption
for caption in tqdm(annotations_value):
    image_id = caption['image_id']
    caption_text = caption['caption']

    # 如果该图片id还未在字典中出现过，则添加到字典中
    if image_id not in id_caption:
        id_caption[image_id] = caption_text

#split后数据的保存，保存至txt文件
split_ratio = 0.8 #seen的图像占比
train_id,test_id = split_dataset(list(id_caption.keys()),split_ratio)
#对seen_caption进行解析并统计三类视觉概念
for key in tqdm(train_id):
  caption = id_caption[key]
  objects_train.update(sen_parse(caption=caption)[0])
  attributes_train.update(sen_parse(caption=caption)[1])
  relations_train.update(sen_parse(caption=caption)[2])
            
unseen_comp_id = []
unseen_atoms_id= []
caption_test = []
#遍历test中的id_caption，并对其进行分类
for key in tqdm(test_id):
  attributes_test = set()
  relations_test = set()  
  objects_test = set()
  caption_test = id_caption[key]
  objects_test.update(sen_parse(caption=caption_test)[0])
  attributes_test.update(sen_parse(caption=caption_test)[1])
  relations_test.update(sen_parse(caption=caption_test)[2])

  if attributes_test.issubset(attributes_train) and relations_test.issubset(relations_train)\
  and (objects_test.issubset(objects_train)):
    unseen_comp_id.append(key)
  else:
    unseen_atoms_id.append(key)#前提test中没有train中图片的不同view,#COCO数据集本身train与test的图像有无交集??



#split后三类数据的保存，保存至txt文件
unseen_atoms_id_path = r"/content/unseen_atoms.txt"
unseen_comp_id_path = r"/content/unseen_comp.txt"
seen_id_path = r"/content/seen.txt"
text = 'image_id'+ '\t'+'caption'+ '\n'

for id in unseen_comp_id:
    text += str(id) + '\t' + id_caption[id] + '\n'
with open(unseen_comp_id_path, 'w') as f:
    f.write(text)


text = 'image_id'+ '\t'+'caption'+ '\n'
for id in unseen_atoms_id:
    text += str(id) + '\t' + id_caption[id] + '\n'
with open(unseen_atoms_id_path, 'w') as f:
    f.write(text)


text = 'image_id'+ '\t'+'caption'+ '\n'
for id in train_id:
    text += str(id) + '\t' + id_caption[id] + '\n'
with open(seen_id_path, 'w') as f:
    f.write(text)