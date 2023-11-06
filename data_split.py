import random
import json
import sng_parser
import spacy
from tqdm import *
import random
from utils import load_config_file,set_seed

def extract_adjectives(sentence):
    # 加载英文语言模型
    nlp = spacy.load('en_core_web_sm')
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

def save_id_caption(path,id_list,id_caption):
  text = 'image_id'+ '\t'+'caption'+ '\n'
  for id in id_list:
      text += str(id) + '\t' + id_caption[id] + '\n'
  with open(path, 'w') as f:
    f.write(text)


def split_dataset(dataset, split_ratio):
    # 设置随机种子为10
    set_seed(seed=10, n_gpu=0)
    random.shuffle(dataset)  # 随机打乱数据集顺序
    split_index = int(len(dataset) * split_ratio)
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]
    return train_data, test_data

def load_id_caption(caption_path):
  with open(caption_path, 'r') as f1:
      dictortary = json.load(f1)

  annotations_value = dictortary.get("annotations")

  id_caption = dict()
  #每个测试图像仅保留一个caption
  for caption in tqdm(annotations_value):
      image_id = caption['image_id']
      caption_text = caption['caption']

      # 如果该图片id还未在字典中出现过，则添加到字典中
      if image_id not in id_caption:
          id_caption[image_id] = caption_text
  return id_caption

def collect_seen_items(id_caption,seen_id):
    attributes_train = set()
    relations_train = set()
    objects_train = set()
    for key in tqdm(seen_id):
      caption = id_caption[key]
      objects_train.update(sen_parse(caption=caption)[0])
      attributes_train.update(sen_parse(caption=caption)[1])
      relations_train.update(sen_parse(caption=caption)[2])

    return objects_train,attributes_train,relations_train

def find_unseen_comp(id_caption,test_id):
    unseen_comp_id = []
    unseen_atoms_id= []
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
        unseen_atoms_id.append(key)
    return unseen_comp_id,unseen_atoms_id
   

if __name__ == '__main__':
    DATA_CONFIG_PATH = "data\data_config.yaml"
    config = load_config_file(DATA_CONFIG_PATH)
    caption_path = config.annotation_caption_file
    id_caption = load_id_caption(caption_path=caption_path)
  
    split_ratio = 0.8 #seen的图像占比
    seen_id,test_id = split_dataset(list(id_caption.keys()),split_ratio)
    #对训练caption进行解析#对seen_caption进行解析并统计三类视觉概念
    objects_train,attributes_train,relations_train = collect_seen_items(id_caption,seen_id)
    unseen_comp_id,_=find_unseen_comp(id_caption,test_id)
                
    #split后三类数据的保存，保存至txt文件
    unseen_comp_id_path = config.unseen_imgid_caption_dir
    seen_id_path = config.seen_imgid_caption_dir
    

    save_id_caption(unseen_comp_id_path,unseen_comp_id,id_caption)
    save_id_caption(seen_id_path,seen_id,id_caption)
