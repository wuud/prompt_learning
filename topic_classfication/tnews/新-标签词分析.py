import jieba
import json
import os
import torch
from sentence_transformers import SentenceTransformer,util

train_file = 'E:\datasets\\tnews\\train.json'
map = {100:"故事", 101:"文化", 102:"娱乐", 103:"体育", 104:"财经", 106:"房产", 107:"汽车", 108:"教育", 109:"科技", 110:"军事", 112:"旅游", 113:"国际", 114:"证券", 115:"农业", 116:"电竞"}
file_map = {"民生":'story.txt', "文化":'culture.txt', "娱乐":'entertainment.txt', "体育":'sports.txt', "财经":'finance.txt', "房产":'house.txt', "汽车":'car.txt', "教育":'edu.txt', "科技":'tech.txt', "军事":'military.txt', "旅游":'travel.txt', "国际":'word.txt', "证券":'stock.txt', "农业":'agriculture.txt', "电竞":'game.txt'}
# 0.5487562189054727
# all_core_words = [['故事','小说'],['文化','文明','诗词','书法','书画'],["娱乐",'音乐','影视','综艺','明星'],["体育",'运动','篮球','足球','排球'],["财经",'金融'],["房产",'房地产'],["汽车",'轿车'],["教育",'教书'],["科技",'技术'],["军事",'军队'],["旅游",'旅行'],["国际",'世界'],["证券",'股票'],["农业",'农学'],["电竞",'游戏']]
# core_words2
all_core_words = [['故事'],['文化'],["娱乐"],["体育"],["财经"],["房产"],["汽车"],["教育"],["科技"],["军事"],["旅游"],["国际"],["证券"],["农业"],["电竞"]]

classes = ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游", "国际", "证券", "农业", "游戏"]
FREQ = 3
SCORE = 0

'''
此方案思路：对每个核心词单独进行语义搜索，然后将得到的词合并去重
'''
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

def list_to_set(s, l):
    for e in l:
        s.add(e)

# 传入的是embedding的list
def get_embeddings_avg(embeddings):
    res = torch.zeros(embeddings[0].shape).to('cuda')
    # print(res)
    # print(embeddings[0])
    for embedding in embeddings:
        res += embedding

    return res/len(embeddings)

# 得到词语和所有核心之间最大的那个语义相似度
def get_max_sim(word_embedding, core_word_embeddings):
    score = 0
    for core_word_embedding in core_word_embeddings:
        score = max(score, util.pytorch_cos_sim(word_embedding, core_word_embedding))
    return score

get_data_set(train_file)

# for top_k in [40, 50, 60, 70, 80, 90, 100]:
# datasets key 是 label， value是list。list中每个元素都是句子
# all_counts是dict， key是label， value是每个label对应的分词结果也是dict

filter_words = []
filter_embeddings = []
filter_words_dict = []
words_length = {}
# 筛选语义相似度大于score，并且词频大于freq的词
def get_first_filter_words(freq, score):
    global filter_words
    global filter_embeddings
    if len(filter_words) > 0:
        return filter_words
    for idx, dataset in enumerate(datasets.items()):
        counts = {}
        after_dict = {}
        res = []
        print('=' * 100, map[dataset[0]], '=' * 100)
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
            word_freq = item[1]
            if word_freq <= freq:
                continue

            word_embedding = model.encode(word, convert_to_tensor=True)
            score_num = util.pytorch_cos_sim(core_embedding_avg, word_embedding)
            # if score_num > score:
            res.append(word)
            word_embeddings.append(word_embedding)
            after_dict[word] = word_freq
        filter_words.append(res)
        filter_embeddings.append(word_embeddings)
        filter_words_dict.append(after_dict)
        print('after filter: ', len(res))
        words_length[map[dataset[0]]] = len(res)
        assert len(res) == len(word_embeddings)
    return filter_words, filter_embeddings

# 根据语义相似度进行去重
# 把筛选后的词语再和其他标签词（核心词）计算语义相似度，如果词语与当前核心词的语义相似度大于其他所有核心词的语义相似度则留下，否则去除。
# all_words 是list，里面每个元素是set
def second_filter(all_words):
    res = []
    for idx, words in enumerate(all_words):
        core_words = all_core_words[idx]
        core_word_embeddings = trans_to_embedding(core_words)
        del_words = set()
        for word in words:
            word_embedding = model.encode(word, convert_to_tensor=True)
            best_score = get_max_sim(word_embedding, core_word_embeddings)
            for idy, core_words in enumerate(all_core_words):
                if idy == idx:
                    continue
                score = get_max_sim(word_embedding, trans_to_embedding(core_words))
                if score >= best_score:
                    del_words.add(word)
        res.append(words - del_words)
        print('删除{}条词语，{}'.format(len(del_words), str(del_words)))

    return res

def filter_by_sim(core_embdding, word_embeddings):
    res = []
    for word_embedding in word_embeddings:
        sim = util.pytorch_cos_sim(core_embdding, word_embedding)
        if sim >= SCORE:
            res.append(word_embedding)
    return res



# 开始之前先把所有的初次筛选的词放入全局list变量中
get_first_filter_words(FREQ, SCORE)
print(filter_words)
print(words_length)

# 由于每个类的训练集数据并不均等，所以要对数据较多的类选取更多的标签词
# top_k_list = [int(item[1] / 3) for item in words_length.items()]
# print(top_k_list)
final_res = []
str_res = ''
for i in range(len(classes)):
    print('=' * 50, classes[i], '=' * 50)
    label_words = filter_words[i]
    words_embedding = filter_embeddings[i]
    core_words = all_core_words[i]
    words_dict = filter_words_dict[i]
    word_set = set()
    filter_dict = {}
    # sorted_final_words = []
    for core_word in core_words:
        core_word_embedding = model.encode(core_word, convert_to_tensor=True)
        # 在进行语义搜索前，对语义相似度小于SCORE的词进行过滤
        words_embedding = filter_by_sim(core_word_embedding, words_embedding)
        # 进行语义搜索
        search_res = util.semantic_search(query_embeddings=core_word_embedding, corpus_embeddings=words_embedding,
                                          top_k=top_k)
        # 从search_res中拿到词的下标
        word_indexs = []
        final_words = []
        for word_index in search_res[0]:
            index = word_index['corpus_id']
            word_indexs.append(index)
            final_words.append(label_words[index])
        print(word_indexs)
        print('after final words: ', len(final_words))
        list_to_set(word_set, final_words)
    # 根据set中的词，去dict中拿到对应词频
    # for word in word_set:
    #     filter_dict[word] = words_dict[word]
    # 利用词频排序，并进行截断只取topk
    # filter_dict = sorted(filter_dict.items(), key=lambda x: x[1], reverse=True)
    # for item in filter_dict:
    #     sorted_final_words.append(item[0])
    # 将dict的内容写入到最终list
    final_res.append(word_set)
    str_res += trans_list_to_string(word_set)
    str_res += '\n'

# second_words = second_filter(final_res)
# for sec_word_set in second_words:
#     str_res += trans_list_to_string(sec_word_set)
#     str_res += '\n'
#
# print(second_words)
# 将每一次迭代产生的词记录到文件中
with open(f'search_res/111.txt', 'w', encoding='gbk') as f:
    f.write(str_res)
    print(str_res)
    print('写入文件成功！')