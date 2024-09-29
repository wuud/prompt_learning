# encoding=utf-8
import jieba
import json
import os
from pyltp import Segmentor
import torch
from sentence_transformers import SentenceTransformer,util

train_file = 'E:\\datasets\\csldcp\\train_few_all.json'

# core_words1
# all_core_words = [['材料科学与工程', '材料'], ['作物学', '作物'], ['口腔医学', '口腔'], ['药学', '药学'], ['教育学', '教育'], ['水利工程', '水利'], ['理论经济学', '理经'], ['食品科学与工程', '食品'], ['畜牧学', '兽医学', '兽医'], ['体育学', '体育'], ['核科学与技术', '核能'], ['力学', '力学'], ['园艺学', '园艺'], ['水产', '水产'], ['法学', '法学'], ['地质学', '地质资源与地质工程', '地质'], ['石油与天然气工程', '能源'], ['农林经济管理', '农林'], ['信息与通信工程', '通信'], ['图书馆、情报与档案管理', '情报'], ['政治学', '政治'], ['电气工程', '电气'], ['海洋科学', '海洋'], ['民族学', '民族'], ['航空宇航科学与技术', '航空'], ['化学', '化学工程与技术', '化工'], ['哲学', '哲学'], ['公共卫生与预防医学', '卫生'], ['艺术学', '艺术'], ['农业工程', '农工'], ['船舶与海洋工程', '船舶'], ['计算机科学与技术', '计科'], ['冶金工程', '冶金'], ['交通运输工程', '交通'], ['动力工程及工程热物理', '动力'], ['纺织科学与工程', '纺织'], ['建筑学', '建筑'], ['环境科学与工程', '环境'], ['公共管理', '公管'], ['数学', '数学'], ['物理学', '物理'], ['林学/林业工程', '林业'], ['心理学', '心理'], ['历史学', '历史'], ['工商管理', '工商'], ['应用经济学', '应经'], ['中医学','中药学', '中医'], ['天文学', '天文'], ['机械工程', '机械'], ['土木工程', '土木'], ['光学工程', '光学'], ['地理学', '地理'], ['农业资源利用', '农资'], ['生物学', '生物科学与工程', '生物'], ['兵器科学与技术', '兵器'], ['矿业工程', '矿业'], ['大气科学', '大气'], ['基础医学/临床医学', '医学'], ['电子科学与技术', '电子'], ['测绘科学与技术', '测绘'], ['控制科学与工程', '控制'], ['军事学', '军事'], ['中国语言文学', '语言'], ['新闻传播学', '新闻'], ['社会学', '社会'], ['地球物理学', '地球'], ['植物保护', '植物']]
# 0.362
# all_core_words = [['材料科学与工程', '材料'], ['作物学', '作物'], ['口腔医学', '口腔'], ['药学', '药学'], ['教育学', '教育'], ['水利工程', '水利'], ['理论经济学','理论经济','理经'], ['食品科学与工程', '食品'], ['畜牧学', '兽医学', '兽医'], ['体育学', '体育'], ['核科学与技术', '核能'], ['力学', '力学'], ['园艺学', '园艺'], ['水产', '水产'], ['法学', '法学'], ['地质学', '地质资源与地质工程'], ['石油与天然气工程', '能源'], ['农林经济管理', '农林'], ['信息与通信工程', '通信', '信息'], ['图书馆','情报','档案管理'], ['政治学', '政治'], ['电气工程', '电气'], ['海洋科学', '海洋'], ['民族学', '民族'], ['航空','宇航', '航空'], ['化学', '化学工程与技术', '化工'], ['哲学', '哲学'], ['公共卫生','预防医学', '卫生'], ['艺术学', '艺术'], ['农业工程', '农工'], ['船舶','海洋工程'], ['计算机科学与技术', '计算机','计科'], ['冶金工程', '冶金'], ['交通运输工程', '交通'], ['动力工程', '工程热物理', '动力'], ['纺织科学与工程', '纺织'], ['建筑学', '建筑'], ['环境科学与工程', '环境'], ['公共管理', '公管'], ['数学', '数学'], ['物理学', '物理'], ['林学', '林业工程', '林业'], ['心理学', '心理'], ['历史学', '历史'], ['工商管理', '工商'], ['应用经济学','应用经济','应经'], ['中医学','中药学', '中医'], ['天文学', '天文'], ['机械工程', '机械'], ['土木工程', '土木'], ['光学工程', '光学'], ['地理学', '地理'], ['农业资源利用', '农业'], ['生物学', '生物科学与工程', '生物'], ['兵器科学与技术', '兵器'], ['矿业工程', '矿业'], ['大气科学', '大气'], ['基础医学', '临床医学', '医学'], ['电子科学与技术', '电子'], ['测绘科学与技术', '测绘'], ['控制科学与工程', '控制'], ['军事学', '军事'], ['中国语言文学', '语言'], ['新闻传播学', '新闻'], ['社会学', '社会'], ['地球物理学', '地球'], ['植物保护', '植物']]
# core words 3
# all_core_words = [['材料科学与工程', '材料'], ['作物学', '作物'], ['口腔医学', '口腔'], ['药学', '药学'], ['教育学', '教育'], ['水利工程', '水利'], ['理论经济学','理论经济','理经'], ['食品科学与工程', '食品'], ['畜牧学', '兽医学', '兽医'], ['体育学', '体育'], ['核科学与技术', '核能', '核科学' ], ['力学', '力'], ['园艺学', '园艺'], ['水产', '水生'], ['法学', '法律'], ['地质学', '地质资源与地质工程', '地质资源', '地质工程'], ['石油与天然气工程', '能源', '石油', '天然气'], ['农林经济管理', '农林'], ['信息与通信工程', '通信', '信息'], ['图书馆','情报','档案管理'], ['政治学', '政治'], ['电气工程', '电气'], ['海洋科学', '海洋'], ['民族学', '民族'], ['航空','宇航', '航空'], ['化学', '化学工程与技术', '化工'], ['哲学', '哲学'], ['公共卫生','预防医学', '卫生'], ['艺术学', '艺术'], ['农业工程', '农工'], ['船舶','海洋工程'], ['计算机科学与技术', '计算机','计科'], ['冶金工程', '冶金'], ['交通运输工程', '交通'], ['动力工程', '工程热物理', '动力'], ['纺织科学与工程', '纺织'], ['建筑学', '建筑'], ['环境科学与工程', '环境'], ['公共管理', '公管'], ['数学', '数学'], ['物理学', '物理'], ['林学', '林业工程', '林业'], ['心理学', '心理'], ['历史学', '历史'], ['工商管理', '工商'], ['应用经济学','应用经济','应经'], ['中医学','中药学', '中医'], ['天文学', '天文'], ['机械工程', '机械'], ['土木工程', '土木'], ['光学工程', '光学'], ['地理学', '地理'], ['农业资源利用', '农业'], ['生物学', '生物科学与工程', '生物'], ['兵器科学与技术', '兵器'], ['矿业工程', '矿业'], ['大气科学', '大气'], ['基础医学', '临床医学', '医学'], ['电子科学与技术', '电子'], ['测绘科学与技术', '测绘'], ['控制科学与工程', '控制'], ['军事学', '军事'], ['中国语言文学', '语言'], ['新闻传播学', '新闻'], ['社会学', '社会'], ['地球物理学', '地球'], ['植物保护', '植物']]
all_core_words = [['材料科学与工程'], ['作物学'], ['口腔医学'], ['药学'], ['教育学'], ['水利工程'], ['理论经济学'], ['食品科学与工程'], ['畜牧学/兽医学'], ['体育学'], ['核科学与技术'], ['力学'], ['园艺学'], ['水产'], ['法学'], ['地质学/地质资源与地质工程'], ['石油与天然气工程'], ['农林经济管理'], ['信息与通信工程'], ['图书馆、情报与档案管理'], ['政治学'], ['电气工程'], ['海洋科学'], ['民族学'], ['航空宇航科学与技术'], ['化学/化学工程与技术'], ['哲学'], ['公共卫生与预防医学'], ['艺术学'], ['农业工程'], ['船舶与海洋工程'], ['计算机科学与技术'], ['冶金工程'], ['交通运输工程'], ['动力工程及工程热物理'], ['纺织科学与工程'], ['建筑学'], ['环境科学与工程'], ['公共管理'], ['数学'], ['物理学'], ['林学/林业工程'], ['心理学'], ['历史学'], ['工商管理'], ['应用经济学'], ['中医学/中药学'], ['天文学'], ['机械工程'], ['土木工程'], ['光学工程'], ['地理学'], ['农业资源利用'], ['生物学/生物科学与工程'], ['兵器科学与技术'], ['矿业工程'], ['大气科学'], ['基础医学/临床医学'], ['电子科学与技术'], ['测绘科学与技术'], ['控制科学与工程'], ['军事学'], ['中国语言文学'], ['新闻传播学'], ['社会学'], ['地球物理学'], ['植物保护']]

classes = ['材料科学与工程', '作物学', '口腔医学', '药学', '教育学', '水利工程', '理论经济学', '食品科学与工程', '畜牧学/兽医学', '体育学', '核科学与技术', '力学', '园艺学', '水产', '法学', '地质学/地质资源与地质工程', '石油与天然气工程', '农林经济管理', '信息与通信工程', '图书馆、情报与档案管理', '政治学', '电气工程', '海洋科学', '民族学', '航空宇航科学与技术', '化学/化学工程与技术', '哲学', '公共卫生与预防医学', '艺术学', '农业工程', '船舶与海洋工程', '计算机科学与技术', '冶金工程', '交通运输工程', '动力工程及工程热物理', '纺织科学与工程', '建筑学', '环境科学与工程', '公共管理', '数学', '物理学', '林学/林业工程', '心理学', '历史学', '工商管理', '应用经济学', '中医学/中药学', '天文学', '机械工程', '土木工程', '光学工程', '地理学', '农业资源利用', '生物学/生物科学与工程', '兵器科学与技术', '矿业工程', '大气科学', '基础医学/临床医学', '电子科学与技术', '测绘科学与技术', '控制科学与工程', '军事学', '中国语言文学', '新闻传播学', '社会学', '地球物理学', '植物保护']

model_path = 'E://models//chinese-roberta-wwm-ext'
model = SentenceTransformer(model_path)
top_k = 100
datasets = {}
for item in classes:
    datasets[item] = []

with open(train_file, encoding='utf-8') as f:
    for idx, line in enumerate(f):
        line = json.loads(line)
        datasets[line['label']].append(line['content'])

# print(datasets)

def stopwordlist():
    stopwords = [line.strip() for line in open('stop_words.txt', encoding='UTF-8').readlines()]
    # ---停用词补充,视具体情况而定---
    i = 0
    for i in range(19):
        stopwords.append(str(10 + i))
    return stopwords


# # pkuseg分词
# def pkuseg_cut(sent):
#     seg = pkuseg.pkuseg()
#     words = seg.cut(sent)
#     return words

cws_model_path = os.path.join('../data/cws.model') # 分词模型路径，模型名称为`cws.model`
segmentor = Segmentor(cws_model_path)


# 对句子进行分词，并输出前K个词语及其词频
def seg_word(line, counts):
    seg=jieba.cut_for_search(line.strip())
    # seg = jieba.cut(line.strip())
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
        if freq < 2:
            continue

        word_embedding = model.encode(word, convert_to_tensor=True)
        score = util.pytorch_cos_sim(core_embedding_avg, word_embedding)
        if score > 0.7:
            res.append(word)
            word_embeddings.append(word_embedding)
    # word_embeddings_avg = get_embeddings_avg(word_embeddings)
    # 进行语义搜索
    search_res = util.semantic_search(query_embeddings=core_embedding_avg, corpus_embeddings=word_embeddings,
                                      top_k=30)
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