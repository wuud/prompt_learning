import codecs
import json

import torch
from datasets import load_from_disk
from openprompt.data_utils import InputExample


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
    return my_data_set, 256


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

    get_data_set('../../topic_sentiment_classfication/dataset/hotel_train.txt', 'train')
    get_data_set('../../topic_sentiment_classfication/dataset/hotel_test.txt', 'test')
    return my_data_set, 128


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
    return my_data_set, 128