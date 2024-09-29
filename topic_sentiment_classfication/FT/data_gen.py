import codecs
import json

import pandas as pd
import torch
from openprompt.data_utils import InputExample
from openprompt.data_utils.data_sampler import FewShotSampler
from datasets import load_dataset


class HotelDataset(torch.utils.data.Dataset):
    def __init__(self, train_file, val_file, test_file, split):
        datasets = {}
        datasets['train'] = load_dataset("json", data_files=train_file)
        datasets['valid'] = load_dataset("json", data_files=val_file)
        datasets['test'] = load_dataset("json", data_files=test_file)
        self.dataset = datasets[split]['train']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text_a']
        label = int(self.dataset[i]['label'])
        # topic = self.dataset[i]["meta"]['topic']

        return text, label

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

    get_data_set('../dataset/hotel_train.txt', 'train')
    get_data_set('../dataset/hotel_test.txt', 'test')
    return my_data_set


def few_shot_sample(seed, shot, dataset, dataset_name):
    train_file = f'./{dataset_name}/train_res.txt'
    val_file = f'./{dataset_name}/val_res.txt'
    sampler = FewShotSampler(num_examples_per_label=shot, also_sample_dev=True, num_examples_per_label_dev=shot)
    train_dataset, valid_dataset = sampler(dataset['train'], seed=seed)
    train_str = ''
    for i in range(len(train_dataset)):
        dic = train_dataset[i].to_dict()
        train_str += json.dumps(dic, ensure_ascii=False)
        train_str += '\n'
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(train_str)

    valid_str = ''
    for i in range(len(valid_dataset)):
        dic = valid_dataset[i].to_dict()
        valid_str += json.dumps(dic, ensure_ascii=False)
        valid_str += '\n'
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write(valid_str)

    return train_file, val_file

def trans_test_dataset(dataset, dataset_name):
    test_dataset = dataset['test']
    test_file = f'./{dataset_name}/test_res.txt'
    test_str = ''
    for i in range(len(test_dataset)):
        dic = test_dataset[i].to_dict()
        dic_str = json.dumps(dic, ensure_ascii=False)
        test_str += dic_str
        test_str += '\n'
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_str)
    return test_file