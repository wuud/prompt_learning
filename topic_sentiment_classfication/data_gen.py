import torch
from openprompt.data_utils import InputExample
import codecs
import random

'''
酒店数据集的清洗和整理
'''
def gen_hotel_data(src_path, dest_path):
    lines = codecs.open(src_path, encoding='utf-8').readlines()
    data = []
    eles = []
    for line in lines:
        line = line.replace(' ', '')
        line = line.replace(',', '，')
        eles.append(line.strip())
        if len(eles) == 3:
            data.append(eles)
            eles = []

    for eles in data:
        eles[0] = eles[0].replace('$T$', eles[1])

    print(data)
    with open(dest_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(','.join(str(item) for item in line) for line in data))
        file.write('\n')


gen_hotel_data('dataset/raw_Hotel_Test.txt', 'dataset/hotel_test.txt')



'''
BDCI car 汽车数据集的整理和清洗
'''
def gen_car_data():
    # 按十分之一的比例，生成训练集和测试集
    lines = codecs.open("dataset/car_all_data.csv", encoding='utf-8').readlines()
    train_data = {}
    train_data[0] = []
    train_data[1] = []
    train_data[2] = []
    print(len(lines))
    for line in lines[1:]:
        eles = line.split(',')[0:4]
        eles[1] = eles[1].strip()
        eles[1] = eles[1].replace('\xa0', '')
        # print(eles)
        label = int(eles[-1])
        if label == -1:
            train_data[0].append(eles)
        elif label == 0:
            train_data[1].append(eles)
        else:
            train_data[2].append(eles)
    print('all_data:,', len(train_data[0]), len(train_data[1]), len(train_data[2]))
    print(train_data)
    test_data = {}
    for i in range(3):
        num_elements = len(train_data[i]) // 10

        # 使用sample函数随机选择元素
        random_selection = random.sample(train_data[i], num_elements)
        test_data[i] = random_selection
        # 从原始列表中移除选择的元素
        for element in random_selection:
            train_data[i].remove(element)

    print('train_data:,', len(train_data[0]), len(train_data[1]), len(train_data[2]))
    print('test_data:,', len(test_data[0]), len(test_data[1]), len(test_data[2]))

    # 将生成的数据写入文件中

    with open('./dataset/car_train.txt', 'w', encoding='utf-8') as file:
        for i in range(3):
            file.write('\n'.join(','.join(str(item) for item in line) for line in train_data[i]))
            file.write('\n')
    with open('./dataset/car_test.txt', 'w', encoding='utf-8') as file:
        for i in range(3):
            file.write('\n'.join(','.join(str(item) for item in line) for line in test_data[i]))
            file.write('\n')
