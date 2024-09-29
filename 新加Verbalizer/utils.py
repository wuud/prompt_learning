import codecs
import json

import torch
from datasets import load_from_disk
from openprompt.data_utils import InputExample

def get_tnews_data():
    # 定义数据集
    train_file = 'E:\datasets\\few-tnews\\train_few_all.json'
    # train_file = 'E:\datasets\\tnews\\train.json'
    # val_file = 'E:\datasets\\tnews\\dev.json'
    val_file = 'E:\datasets\\tnews\\dev_few_all.json'
    test_file = 'E:\datasets\\few-tnews\\test_public.json'
    max_length = 50

    def trans_label(label):
        map = {100: 0, 101: 1, 102: 2, 103: 3, 104: 4, 106: 5, 107: 6, 108: 7, 109: 8, 110: 9, 112: 10, 113: 11,
               114: 12, 115: 13, 116: 14}
        return map[label]

    my_data_set = {}

    def get_data_set(file, split):
        my_data_set[split] = []
        with open(file, encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = json.loads(line)
                input = InputExample(guid=idx, label=trans_label(int(line['label'])), text_a=line['sentence'])
                my_data_set[split].append(input)

    get_data_set(train_file, 'train')
    get_data_set(val_file, 'validation')
    get_data_set(test_file, 'test')
    classes = ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游", "国际", "证券", "农业", "电竞"]
    return my_data_set, max_length, classes

def get_csldcp_data():
    # 定义数据集
    train_file = 'E:\\datasets\\csldcp\\train_few_all.json'
    # val_file = 'E:\datasets\\tnews\\dev.json'
    val_file = 'E:\datasets\\csldcp\\dev_few_all.json'
    test_file = 'E:\datasets\\csldcp\\test_public.json'
    max_length = 256
    label_to_index = {'材料科学与工程': 0, '作物学': 1, '口腔医学': 2, '药学': 3, '教育学': 4, '水利工程': 5, '理论经济学': 6, '食品科学与工程': 7,
                      '畜牧学/兽医学': 8, '体育学': 9, '核科学与技术': 10, '力学': 11, '园艺学': 12, '水产': 13, '法学': 14,
                      '地质学/地质资源与地质工程': 15, '石油与天然气工程': 16, '农林经济管理': 17, '信息与通信工程': 18, '图书馆、情报与档案管理': 19, '政治学': 20,
                      '电气工程': 21, '海洋科学': 22, '民族学': 23, '航空宇航科学与技术': 24, '化学/化学工程与技术': 25, '哲学': 26, '公共卫生与预防医学': 27,
                      '艺术学': 28, '农业工程': 29, '船舶与海洋工程': 30, '计算机科学与技术': 31, '冶金工程': 32, '交通运输工程': 33, '动力工程及工程热物理': 34,
                      '纺织科学与工程': 35, '建筑学': 36, '环境科学与工程': 37, '公共管理': 38, '数学': 39, '物理学': 40, '林学/林业工程': 41,
                      '心理学': 42, '历史学': 43, '工商管理': 44, '应用经济学': 45, '中医学/中药学': 46, '天文学': 47, '机械工程': 48, '土木工程': 49,
                      '光学工程': 50, '地理学': 51, '农业资源利用': 52, '生物学/生物科学与工程': 53, '兵器科学与技术': 54, '矿业工程': 55, '大气科学': 56,
                      '基础医学/临床医学': 57, '电子科学与技术': 58, '测绘科学与技术': 59, '控制科学与工程': 60, '军事学': 61, '中国语言文学': 62,
                      '新闻传播学': 63, '社会学': 64, '地球物理学': 65, '植物保护': 66}

    my_data_set = {}

    def get_data_set(file, split):
        my_data_set[split] = []
        with open(file, encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = json.loads(line)
                input = InputExample(guid=idx, label=int(label_to_index[line['label']]), text_a=line['content'])
                my_data_set[split].append(input)

    get_data_set(train_file, 'train')
    get_data_set(val_file, 'validation')
    get_data_set(test_file, 'test')
    classes = ['材料科学与工程', '作物学', '口腔医学', '药学', '教育学', '水利工程', '理论经济学', '食品科学与工程', '畜牧学/兽医学', '体育学', '核科学与技术', '力学',
               '园艺学', '水产', '法学', '地质学/地质资源与地质工程', '石油与天然气工程', '农林经济管理', '信息与通信工程', '图书馆、情报与档案管理', '政治学', '电气工程',
               '海洋科学', '民族学', '航空宇航科学与技术', '化学/化学工程与技术', '哲学', '公共卫生与预防医学', '艺术学', '农业工程', '船舶与海洋工程', '计算机科学与技术',
               '冶金工程', '交通运输工程', '动力工程及工程热物理', '纺织科学与工程', '建筑学', '环境科学与工程', '公共管理', '数学', '物理学', '林学/林业工程', '心理学',
               '历史学', '工商管理', '应用经济学', '中医学/中药学', '天文学', '机械工程', '土木工程', '光学工程', '地理学', '农业资源利用', '生物学/生物科学与工程',
               '兵器科学与技术', '矿业工程', '大气科学', '基础医学/临床医学', '电子科学与技术', '测绘科学与技术', '控制科学与工程', '军事学', '中国语言文学', '新闻传播学', '社会学',
               '地球物理学', '植物保护']

    return my_data_set, max_length, classes

def get_cnews_data():
    # 定义数据集
    train_file = 'E:\datasets\cnews\\cnews.train.txt'
    test_file = "E:\datasets\cnews\\cnews.test.txt"

    map = {'体育': 0, '财经': 1, '房产': 2, '家居': 3, '教育': 4, '科技': 5, '时尚': 6, '时政': 7, '游戏': 8, '娱乐': 9}
    my_data_set = {}
    max_length = 512
    classes = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    def get_data_set(file, split):
        my_data_set[split] = []
        with open(file, encoding='utf-8') as f:
            for idx, line in enumerate(f):
                label = line[:2]
                text = line[3:]
                input = InputExample(guid=idx, label=map[label], text_a=text)
                my_data_set[split].append(input)

    get_data_set(train_file, 'train')
    get_data_set(test_file, 'test')
    return my_data_set, max_length, classes

senti_classes = ['消极', '积极']

def get_chn_data():
    data_path = 'E:\workspace\python\Huggingface_Toturials-main/data/ChnSentiCorp'
    # 定义数据集
    class Dataset(torch.utils.data.Dataset):

        def __init__(self, split):
            self.dataset = load_from_disk(data_path)
            # self.dataset = load_dataset("seamew/ChnSentiCorp") # huggingface 路径
            self.dataset = self.dataset[split]

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, i):
            text = self.dataset[i]['text']
            label = self.dataset[i]['label']

            return text, label

    my_data_set = {}
    for split in ['train', 'validation', 'test']:
        my_data_set[split] = []
        for idx, (text, label) in enumerate(Dataset(split)):
            input = InputExample(guid=idx, label=label, text_a=text)
            my_data_set[split].append(input)
            # print(input)
    return my_data_set, 256, senti_classes


def get_hotel_data():
    my_data_set = {}

    def get_data_set(file, split):
        my_data_set[split] = []
        lines = codecs.open(file, encoding='utf-8').readlines()
        for idx, line in enumerate(lines):
            eles = line.split(',')
            # print(eles[0], eles[1], int(eles[3]))
            input = InputExample(guid=idx, label=0 if int(eles[2]) == -1 else 1, text_a=eles[0], meta={'key': eles[1]})
            my_data_set[split].append(input)
            # print(input)

    get_data_set('../topic_sentiment_classfication/dataset/hotel_train.txt', 'train')
    get_data_set('../topic_sentiment_classfication/dataset/hotel_test.txt', 'test')
    return my_data_set, 128, senti_classes


def get_epr_data():
    train_file = 'E:/datasets/eprstmt/train_few_all.json'
    test_file = 'E:/datasets/eprstmt/test_public.json'

    my_data_set = {}

    def get_data_set(file, split):
        my_data_set[split] = []
        with open(file, encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = json.loads(line)
                input = InputExample(guid=idx, label=1 if line['label'] == 'Positive' else 0, text_a=line['sentence'])
                my_data_set[split].append(input)

    get_data_set(train_file, 'train')
    get_data_set(test_file, 'test')
    return my_data_set, 128, senti_classes
