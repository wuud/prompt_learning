import jieba
import json
import os
import torch
from sentence_transformers import SentenceTransformer,util
import matplotlib.pyplot as plt

train_file = 'E:\datasets\\tnews\\train.json'
map = {100:"民生", 101:"文化", 102:"娱乐", 103:"体育", 104:"财经", 106:"房产", 107:"汽车", 108:"教育", 109:"科技", 110:"军事", 112:"旅游", 113:"国际", 114:"证券", 115:"农业", 116:"电竞"}
file_map = {"民生":'story.txt', "文化":'culture.txt', "娱乐":'entertainment.txt', "体育":'sports.txt', "财经":'finance.txt', "房产":'house.txt', "汽车":'car.txt', "教育":'edu.txt', "科技":'tech.txt', "军事":'military.txt', "旅游":'travel.txt', "国际":'word.txt', "证券":'stock.txt', "农业":'agriculture.txt', "电竞":'game.txt'}
# core_words1
all_core_words = [['故事','小说'],['文化','文明'],["娱乐"],["体育",'运动'],["财经",'金融'],["房产",'房地产'],["汽车"],["教育"],["科技",'技术'],["军事"],["旅游",'旅行'],["国际",'世界'],["证券",'股票'],["农业"],["电竞",'游戏']]
# core_words2
# all_core_words = [['故事'],['文化'],["娱乐"],["体育"],["财经"],["房产"],["汽车"],["教育"],["科技"],["军事"],["旅游"],["国际"],["证券"],["农业"],["电竞"]]
# core_words3
# all_core_words = [['故事','小说'],['文化','文明'],["娱乐",'休闲'],["体育",'运动'],["财经",'金融'],["房产",'房地产'],["汽车",'轿车'],["教育",'教学'],["科技",'技术'],["军事", '军队'],["旅游",'旅行'],["国际",'世界'],["证券",'股票'],["农业", '农学'],["电竞",'游戏']]

classes = ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游", "国际", "证券", "农业", "电竞"]

model_path = 'E://models//chinese-roberta-wwm-ext'
model = SentenceTransformer(model_path)
top_k = 40
datasets = {}
for item in map.items():
    datasets[item[0]] = []

def get_data_set(file):
    with open(file, encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            datasets[int(line['label'])].append(line['sentence'])

def stopwordlist():
    stopwords = [line.strip() for line in open('stop_words.txt', encoding='UTF-8').readlines()]
    # ---停用词补充,视具体情况而定---
    i = 0
    for i in range(19):
        stopwords.append(str(10 + i))
    return stopwords


# 对句子进行分词，并输出前K个词语及其词频
def seg_word(line, counts):
    # seg=jieba.cut_for_search(line.strip())
    seg = jieba.cut(line.strip())
    wordstop = stopwordlist()
    for word in seg:
        if word not in wordstop:
            if word != ' ':
                counts[word] = counts.get(word, 0) + 1 #统计每个词出现的次数
    # return  temp #显示分词结果
    # return str(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20])  # 统计出现前二十最多的词及次数

def trans_list_to_string(list):
    res = ''
    for item in list:
        res += str(item)
        res += ','
    return res[:-1]

def trans_to_embedding(words):
    res = []
    for word in words:
        embedding = model.encode(word, convert_to_tensor=True)
        res.append(embedding)
    return res

# 传入的是embedding的list
def get_embeddings_avg(embeddings):
    res = torch.zeros(embeddings[0].shape).to('cuda')
    # print(res)
    # print(embeddings[0])
    for embedding in embeddings:
        res += embedding

    return res/len(embeddings)

get_data_set(train_file)

# 选择 教育科技军事 类别下的数据分析，key是108，109，110
edu_freqs = []
tec_freqs = []
mil_freqs = []
res = {}
for index in [108, 109, 110]:
    dataset = datasets[index]
    res[index] = [0, 0, 0]
    counts = {}
    print('dataset length : ', len(dataset))
    for line in dataset:
        seg_word(line, counts)
    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    print('count length :', len(counts))
    nums = [0 for i in range(10)]

    # 拿到筛选后的词语
    for item in counts:
        tot_score = 0
        word = item[0]
        freq = item[1]
        if freq == 1:
            res[index][0] += 1
        elif freq == 2:
            res[index][1] += 1
        else:
            res[index][2] += 1


print(res)


