import torch
import pandas as pd


l1 = 0  # 长度小于10
l2 = 0  # 10 - 50
l3 = 0  # 50 - 256
l4 = 0  # 256 - 500
l5 = 0  # 500 - 1000
l6 = 0  # > 1000


def count(text):
    global l1
    global l2
    global l3
    global l4
    global l5
    global l6
    length = len(text)
    if length < 10:
        l1 += 1
    elif length < 50:
        l2 += 1
    elif length < 256:
        l3 += 1
    elif length < 500:
        l4 += 1
    elif length < 1000:
        l5 += 1
    else:
        l6 += 1

data_path = 'E:\datasets\weibo_senti_100k\weibo_senti_100k.csv'

dt = pd.read_csv(data_path, encoding='utf-8')

#定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        # data_path = 'E:\datasets\weibo_senti_100k\weibo_senti_100k.csv'
        data_path = 'E:\datasets\waimai_10k\waimai_10k.csv'
        dt = pd.read_csv(data_path, encoding='utf-8')

        self.dataset = dt.values

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i][1]
        label = self.dataset[i][0]

        return text, label


data = Dataset()
print(len(data))
for i in range(len(data)):
    count(data[i][0])

print(l1, l2, l3, l4, l5, l6)





























