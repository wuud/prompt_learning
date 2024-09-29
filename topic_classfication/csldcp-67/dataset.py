import json

from openprompt.plms import load_plm
from openprompt.prompts import MixedTemplate, ManualVerbalizer,ManualTemplate,ProtoVerbalizer
from openprompt import PromptDataLoader
from openprompt import PromptForClassification
from datasets import load_dataset, load_from_disk
from openprompt.data_utils import InputExample
from transformers import AdamW
import torch
import matplotlib.pyplot as plt
from openprompt.data_utils.data_sampler import FewShotSampler

#定义数据集
train_file = 'E:\\datasets\\csldcp\\train_few_all.json'
# val_file = 'E:\datasets\\tnews\\dev.json'
val_file = 'E:\datasets\\csldcp\\dev_few_all.json'
test_file = 'E:\datasets\\csldcp\\test_public.json'



my_data_set = {}
def get_data_set(file, split):
    my_data_set[split] = []
    with open(file, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = json.loads(line)
            input = InputExample(guid=idx, label=int(line['label']), text_a=line['sentence'])
            my_data_set[split].append(input)
get_data_set(train_file, 'train')
get_data_set(val_file, 'validation')
get_data_set(test_file, 'test')

print(len(my_data_set['train']))
print(len(my_data_set['test']))