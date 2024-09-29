import jieba
import json
import os
import torch
from sentence_transformers import SentenceTransformer,util

train_file = 'E:\datasets\\tnews\\train.json'
map = {100:"民生", 101:"文化", 102:"娱乐", 103:"体育", 104:"财经", 106:"房产", 107:"汽车", 108:"教育", 109:"科技", 110:"军事", 112:"旅游", 113:"国际", 114:"证券", 115:"农业", 116:"电竞"}
file_map = {"民生":'story.txt', "文化":'culture.txt', "娱乐":'entertainment.txt', "体育":'sports.txt', "财经":'finance.txt', "房产":'house.txt', "汽车":'car.txt', "教育":'edu.txt', "科技":'tech.txt', "军事":'military.txt', "旅游":'travel.txt', "国际":'word.txt', "证券":'stock.txt', "农业":'agriculture.txt', "电竞":'game.txt'}
# core_words1
all_core_words = [['故事','小说'],['文化','文明'],["娱乐"],["体育",'运动'],["财经",'金融'],["房产",'房地产'],["汽车"],["教育"],["科技",'技术'],["军事"],["旅游",'旅行'],["国际",'世界'],["证券",'股票'],["农业"],["电竞",'游戏']]
# core_words2
# all_core_words = [['故事'],['文化'],["娱乐"],["体育"],["财经"],["房产"],["汽车"],["教育"],["科技"],["军事"],["旅游"],["国际"],["证券"],["农业"],["电竞"]]
# core_words3
# all_core_words = [['故事','小说'],['文化','文明'],["娱乐",'休闲'],["体育",'运动'],["财经",'金融'],["房产",'房地产'],["汽车",'轿车'],["教育",'教学'],["科技",'技术'],["军事", '军队'],["旅游",'旅行'],["国际",'世界'],["证券",'股票'],["农业", '农学'],["电竞",'游戏']]

classes = ["民生", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游", "国际", "证券", "农业", "电竞"]

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

# for top_k in [40, 50, 60, 70, 80, 90, 100]:
# datasets key 是 label， value是list。list中每个元素都是句子
# all_counts是dict， key是label， value是每个label对应的分词结果也是dict
all_words = ''
for idx, dataset in enumerate(datasets.items()):
    counts = {}
    res = []
    print('=' * 100, map[dataset[0]], '=' * 100)
    print(len(dataset[1]))
    # print(dataset[1][:100])
    # all_counts[dataset[0]] = {}
    for line in dataset[1]:
        seg_word(line, counts)
    # counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    # print(json.dumps(counts))
    core_words = all_core_words[idx]
    core_embeddings = trans_to_embedding(core_words)
    core_embedding_avg = get_embeddings_avg(core_embeddings)
    # file_name = file_map[classes[idx]]
    print('before filter : ', len(counts))
    word_embeddings = []
    # 拿到筛选后的词语
    for item in counts.items():
        tot_score = 0
        word = item[0]
        freq = item[1]
        if freq < 4:
            continue

        word_embedding = model.encode(word, convert_to_tensor=True)
        score = util.pytorch_cos_sim(core_embedding_avg, word_embedding)
        if score > 0.7:
            res.append(word)
            word_embeddings.append(word_embedding)
    # word_embeddings_avg = get_embeddings_avg(word_embeddings)
    # 进行语义搜索
    search_res = util.semantic_search(query_embeddings=core_embedding_avg, corpus_embeddings=word_embeddings,
                                      top_k=top_k)
    # 从search_res中拿到词的下标
    word_indexs = []
    final_words = []
    for word_index in search_res[0]:
        index = word_index['corpus_id']
        word_indexs.append(index)
        final_words.append(res[index])
    print(word_indexs)

    print('after filter: ', len(res))
    print('after final words: ', len(final_words))
    # print(res)
    # res_file = './res/' + file_map[map[dataset[0]]]
    all_words += trans_list_to_string(final_words)
    all_words += '\n'
    # with open(res_file, 'w', encoding='utf-8') as f:
    #     f.write(str(final_words))

# print(all_counts)
with open(f'res/111.txt', 'w', encoding='utf-8') as f:
    f.write(all_words)
    print('写入文件成功！')