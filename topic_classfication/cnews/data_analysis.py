# encoding=utf-8
import jieba
import json
import os
from pyltp import Segmentor
import torch
from sentence_transformers import SentenceTransformer,util

train_file = 'E:\datasets\cnews\\cnews.train.txt'

# core words1
all_core_words = [['体育'], ['财经'], ['房产'], ['家居'], ['教育'], ['科技'], ['时尚'], ['时政'], ['游戏'], ['娱乐']]
# core words 4
# all_core_words = [['体育','运动'], ['财经','金融'], ['房产','房地产'], ['家居'], ['教育'], ['科技'], ['时尚'], ['时政','政治'], ['游戏'], ['娱乐']]
# core word2
# all_core_words = [['体育','运动'], ['财经','金融'], ['房产','房地产'], ['家居'], ['教育', '教书', '育人'], ['科技','科学','技术'], ['时尚','流行'], ['时政','政治','时事'], ['游戏','电竞','电子游戏'], ['娱乐']]
# all_core_words = [['体育','运动'], ['财经','金融'], ['房产','房地产'], ['家居'], ['教育', '教书', '育人'], ['科技','科学','技术'], ['时尚','流行'], ['时政','政治','时事'], ['游戏','电子游戏'], ['娱乐']]
# core word3
# all_core_words = [['体育','运动'], ['财经','金融'], ['房产','房地产'], ['家居','装修','居住'], ['教育', '教书', '育人'], ['科技','科学','技术'], ['时尚','流行'], ['时政','政治','时事'], ['游戏','电竞','电子游戏'], ['娱乐','休闲']]

classes = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
map = {'体育':0, '财经':1, '房产':2, '家居':3, '教育':4, '科技':5, '时尚':6, '时政':7, '游戏':8, '娱乐':9}

model_path = 'E://models//chinese-roberta-wwm-ext'
model = SentenceTransformer(model_path)

top_k = 1000
# 待尝试：（5，0.7）（6, 0.7）（5,0.75）（6,0.75）
FREQ = 8   # 筛掉词频小于FREQ的词语
SCORE = 0.7 # 筛除语义相似度小于SCORE的词语

datasets = {}
for item in classes:
    datasets[item] = []

with open(train_file, encoding='utf-8') as f:
    for idx, line in enumerate(f):
        label = line[:2]
        text = line[3:]
        datasets[label].append(text)

# print(len(datasets['体育']))

def stopwordlist():
    stopwords = [line.strip() for line in open('stop_words.txt', encoding='UTF-8').readlines()]
    # ---停用词补充,视具体情况而定---
    i = 0
    for i in range(19):
        stopwords.append(str(10 + i))
    return stopwords


cws_model_path = os.path.join('../data/cws.model') # 分词模型路径，模型名称为`cws.model`
segmentor = Segmentor(cws_model_path)


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
    return str(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20])  # 统计出现前二十最多的词及次数

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


# for top_k in [40, 50, 60, 70, 80, 90, 100]:
# datasets key 是 label， value是list。list中每个元素都是句子
# all_counts是dict， key是label， value是每个label对应的分词结果也是dict
all_words = ''
for idx, dataset in enumerate(datasets.items()):
    counts = {}
    res = []
    print('=' * 100, dataset[0], '=' * 100)
    print(len(dataset[1]))
    # print(dataset[1][:100])
    # all_counts[dataset[0]] = {}
    for line in dataset[1]:
        seg_word(line, counts)
    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    # print(json.dumps(counts))
    core_words = all_core_words[idx]
    core_embeddings = trans_to_embedding(core_words)
    core_embedding_avg = get_embeddings_avg(core_embeddings)
    # file_name = file_map[classes[idx]]
    print('before filter : ', len(counts))
    word_embeddings = []
    # 拿到筛选后的词语
    for item in counts:
        tot_score = 0
        word = item[0]
        freq = item[1]
        if freq < FREQ:
            continue

        word_embedding = model.encode(word, convert_to_tensor=True)
        score = util.pytorch_cos_sim(core_embedding_avg, word_embedding)
        if score > SCORE:
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

print(all_words)
with open(f'res/111.txt', 'w', encoding='utf-8') as f:
    f.write(all_words)
    print('写入文件成功！')