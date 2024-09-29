import jieba
from sentence_transformers import SentenceTransformer,util
import torch
import json

model_path = 'E://models//chinese-roberta-wwm-ext'
model = SentenceTransformer(model_path)

def get_tnews_data():
    train_file = 'E:\datasets\\tnews\\train.json'
    map = {100: "故事", 101: "文化", 102: "娱乐", 103: "体育", 104: "财经", 106: "房产", 107: "汽车", 108: "教育", 109: "科技", 110: "军事",
           112: "旅游", 113: "国际", 114: "证券", 115: "农业", 116: "电竞"}

    all_core_words = [['故事','小说'],['文化','文明','诗词','书法','书画'],["娱乐",'音乐','影视','综艺','明星'],["体育",'运动','篮球','足球','排球'],["财经",'金融'],["房产",'房地产'],["汽车",'轿车'],["教育",'教书'],["科技",'技术'],["军事",'军队'],["旅游",'旅行'],["国际",'世界'],["证券",'股票'],["农业",'农学'],["电竞",'游戏']]


    datasets = {}
    for item in map.items():
        datasets[item[0]] = []

    with open(train_file, encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            datasets[int(line['label'])].append(line['sentence'])
    return datasets, all_core_words

def get_cnews_data():
    train_file = 'E:\datasets\cnews\\cnews.train.txt'
    # core words1
    # all_core_words = [['体育'], ['财经'], ['房产'], ['家居'], ['教育'], ['科技'], ['时尚'], ['时政'], ['游戏'], ['娱乐']]#
    # core words 2 top300: 0.743 0.7582
    # all_core_words = [['体育','运动'], ['财经','金融'], ['房产','房地产'], ['家居'], ['教育', '教书'], ['科技','技术'], ['时尚','流行'], ['时政','政治'], ['游戏','电竞'], ['娱乐','音乐','影视','综艺']]
    all_core_words = [['体育', '运动'], ['财经', '金融'], ['房产', '房地产'], ['家居'], ['教育', '教书'], ['科技', '技术'], ['时尚', '流行'],
                      ['时政', '政治'], ['游戏', '电竞'], ['娱乐', '音乐', '影视', '综艺']]

    classes = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    map = {'体育': 0, '财经': 1, '房产': 2, '家居': 3, '教育': 4, '科技': 5, '时尚': 6, '时政': 7, '游戏': 8, '娱乐': 9}

    datasets = {}
    for item in classes:
        datasets[item] = []

    with open(train_file, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            label = line[:2]
            text = line[3:]
            datasets[label].append(text)
    print(len(datasets['体育']))
    return datasets, all_core_words

def get_csldcp_data():
    # 定义数据集
    train_file = 'E:\\datasets\\csldcp\\train_few_all.json'
    # all_core_words = [['材料科学与工程', '材料'], ['作物学', '作物', '农作物'], ['口腔医学', '口腔','嘴巴'], ['药学', '药学','中药','西药'], ['教育学', '教育','教师'], ['水利工程', '水利'],
    #                   ['理论经济学', '理论经济', '理经'], ['食品科学与工程', '食品','食物'], ['畜牧学', '兽医学', '兽医','动物', '牲畜'], ['体育学', '体育','运动','打球'],
    #                   ['核科学与技术', '核能', '核科学', '原子核'], ['力学', '力'], ['园艺学', '园艺'], ['水产', '水生'], ['法学', '法律'],
    #                   ['地质学', '地质资源与地质工程', '地质资源', '地质工程'], ['石油与天然气工程', '能源', '石油', '天然气'], ['农林经济管理', '农林'],
    #                   ['信息与通信工程', '通信', '信息'], ['图书馆', '情报', '档案管理'], ['政治学', '政治'], ['电气工程', '电气'], ['海洋科学', '海洋'],
    #                   ['民族学', '民族'], ['航空', '宇航', '航空'], ['化学', '化学工程与技术', '化工'], ['哲学', '哲学'], ['公共卫生', '预防医学', '卫生'],
    #                   ['艺术学', '艺术'], ['农业工程', '农工'], ['船舶', '海洋工程'], ['计算机科学与技术', '计算机', '计科','计算'], ['冶金工程', '冶金'],
    #                   ['交通运输工程', '交通'], ['动力工程', '工程热物理', '动力'], ['纺织科学与工程', '纺织'], ['建筑学', '建筑'], ['环境科学与工程', '环境'],
    #                   ['公共管理', '公管'], ['数学', '数学'], ['物理学', '物理'], ['林学', '林业工程', '林业'], ['心理学', '心理'], ['历史学', '历史'],
    #                   ['工商管理', '工商'], ['应用经济学', '应用经济', '应经'], ['中医学', '中药学', '中医'], ['天文学', '天文'], ['机械工程', '机械'],
    #                   ['土木工程', '土木'], ['光学工程', '光学'], ['地理学', '地理'], ['农业资源利用', '农业'], ['生物学', '生物科学与工程', '生物'],
    #                   ['兵器科学与技术', '兵器'], ['矿业工程', '矿业'], ['大气科学', '大气'], ['基础医学', '临床医学', '医学'], ['电子科学与技术', '电子'],
    #                   ['测绘科学与技术', '测绘'], ['控制科学与工程', '控制'], ['军事学', '军事'], ['中国语言文学', '语言'], ['新闻传播学', '新闻'],
    #                   ['社会学', '社会'], ['地球物理学', '地球'], ['植物保护', '植物']]
    all_core_words = [['材料科学与工程'], ['作物学'], ['口腔医学'], ['药学'], ['教育学'], ['水利工程'], ['理论经济学'], ['食品科学与工程'], ['畜牧学/兽医学'],
                      ['体育学'], ['核科学与技术'], ['力学'], ['园艺学'], ['水产'], ['法学'], ['地质学/地质资源与地质工程'], ['石油与天然气工程'], ['农林经济管理'],
                      ['信息与通信工程'], ['图书馆、情报与档案管理'], ['政治学'], ['电气工程'], ['海洋科学'], ['民族学'], ['航空宇航科学与技术'], ['化学/化学工程与技术'],
                      ['哲学'], ['公共卫生与预防医学'], ['艺术学'], ['农业工程'], ['船舶与海洋工程'], ['计算机科学与技术'], ['冶金工程'], ['交通运输工程'],
                      ['动力工程及工程热物理'], ['纺织科学与工程'], ['建筑学'], ['环境科学与工程'], ['公共管理'], ['数学'], ['物理学'], ['林学/林业工程'],
                      ['心理学'], ['历史学'], ['工商管理'], ['应用经济学'], ['中医学/中药学'], ['天文学'], ['机械工程'], ['土木工程'], ['光学工程'],
                      ['地理学'], ['农业资源利用'], ['生物学/生物科学与工程'], ['兵器科学与技术'], ['矿业工程'], ['大气科学'], ['基础医学/临床医学'], ['电子科学与技术'],
                      ['测绘科学与技术'], ['控制科学与工程'], ['军事学'], ['中国语言文学'], ['新闻传播学'], ['社会学'], ['地球物理学'], ['植物保护']]

    classes = ['材料科学与工程', '作物学', '口腔医学', '药学', '教育学', '水利工程', '理论经济学', '食品科学与工程', '畜牧学/兽医学', '体育学', '核科学与技术', '力学',
               '园艺学', '水产', '法学', '地质学/地质资源与地质工程', '石油与天然气工程', '农林经济管理', '信息与通信工程', '图书馆、情报与档案管理', '政治学', '电气工程',
               '海洋科学', '民族学', '航空宇航科学与技术', '化学/化学工程与技术', '哲学', '公共卫生与预防医学', '艺术学', '农业工程', '船舶与海洋工程', '计算机科学与技术',
               '冶金工程', '交通运输工程', '动力工程及工程热物理', '纺织科学与工程', '建筑学', '环境科学与工程', '公共管理', '数学', '物理学', '林学/林业工程', '心理学',
               '历史学', '工商管理', '应用经济学', '中医学/中药学', '天文学', '机械工程', '土木工程', '光学工程', '地理学', '农业资源利用', '生物学/生物科学与工程',
               '兵器科学与技术', '矿业工程', '大气科学', '基础医学/临床医学', '电子科学与技术', '测绘科学与技术', '控制科学与工程', '军事学', '中国语言文学', '新闻传播学', '社会学',
               '地球物理学', '植物保护']

    datasets = {}
    for item in classes:
        datasets[item] = []

    with open(train_file, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = json.loads(line)
            datasets[line['label']].append(line['content'])

    return datasets, all_core_words


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