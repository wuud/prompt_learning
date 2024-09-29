
import json
from openprompt.data_utils import InputExample


#定义数据集
train_file = 'E:\datasets\cnews\\cnews.train.txt'
val_file = 'E:\datasets\\cnews\\cnews.val.txt'
test_file = 'E:\datasets\\cnews\\cnews.test.txt'



# my_data_set = {}
# def get_data_set(file, split):
#     my_data_set[split] = []
#     with open(file, encoding='utf-8') as f:
#         for idx, line in enumerate(f):
#             line = json.loads(line)
#             input = InputExample(guid=idx, label=int(line['label']), text_a=line['sentence'])
#             my_data_set[split].append(input)
# get_data_set(train_file, 'train')
# get_data_set(val_file, 'validation')
# get_data_set(test_file, 'test')

my_data_set = {}
def get_data_set(file, split):
    my_data_set[split] = []
    with open(file, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            label = line[:2]
            text = line[3:]
            input = InputExample(guid=idx, label=label, text_a=text)
            my_data_set[split].append(input)
get_data_set(train_file, 'train')
# get_data_set(val_file, 'validation')
get_data_set(test_file, 'test')
print(len(my_data_set['train']))
# print(len(my_data_set['validation']))
print(len(my_data_set['test']))