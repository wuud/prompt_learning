# coding=utf-8

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
import json

model_path = 'E://models//chinese-roberta-wwm-ext'
batch_size = 4
EPOCH = 20
max_length = 512
device = 'cuda'
shot = 16
learning_rate = 5e-1

# 定义 PLM
plm, tokenizer, model_config, wrapper_class = load_plm("bert", model_path) # 本地路径
# plm, tokenizer, model_config, wrapper_class = load_plm("bert", 'bert-base-chinese') # huggingface 仓库

#定义数据集
train_file = 'E:\datasets\\few-tnews\\train_few_all.json'
# val_file = 'E:\datasets\\tnews\\dev.json'
val_file = 'E:\datasets\\tnews\\dev_few_all.json'
test_file = 'E:\datasets\\few-tnews\\test_public.json'
def trans_label(label):
    map = {100:0, 101:1, 102:2, 103:3, 104:4, 106:5, 107:6, 108:7, 109:8, 110:9, 112:10, 113:11, 114:12, 115:13, 116:14}
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

print('train dataset length: ', len(my_data_set['train']))
# print(my_data_set['train'])
print('validation dataset length: ', len(my_data_set['validation']))

def evaluate(data_loader, type='validation'):
    all_preds = []
    all_labels = []
    for step, inputs in enumerate(data_loader):
        inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        # print('-' * 100)
        # print('labels is : ', all_labels)
        # print('prediction is : ',all_preds)
    acc = sum([int(i == j) for i, j in zip(all_preds, all_labels)]) / len(all_preds)
    print(type, ' accuracy: ', acc)
    return acc

def trans_list_to_string(list):
    res = ''
    for item in list:
        res += str(item)
        res += ','
    return res[:-1]
def process_labels(k):
    res = ''
    with open('./verbalizer_300.txt', 'r', encoding='gbk') as f:
        for line in f:
            words = line.split(',')
            # print(len(words))
            if len(words) > k:
                res += trans_list_to_string(words[:k])
                res += '\n'
            else:
                res += trans_list_to_string(words)
    print(res)
    with open('./verbalizer.txt', 'w', encoding='gbk') as f:
        f.write(res)

prompt_template = ManualTemplate(text='{"placeholder":"text_a" }。好好思考一下，这包括{"mask"}。', tokenizer=tokenizer) # template1

test_dataloader = PromptDataLoader(dataset=my_data_set['test'], template=prompt_template, tokenizer=tokenizer,
    tokenizer_wrapper_class=wrapper_class, max_seq_length=max_length, decoder_max_length=3,
    batch_size=batch_size,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

classes = ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游", "国际", "证券", "农业", "电竞"]

acc_nums = []
word_nums = []
for i in range(10, 300, 10):
    print('+' * 50, i, '+' * 50)
    process_labels(i)
    prompt_verbalizer = ManualVerbalizer(classes=classes, tokenizer=tokenizer).from_file('./verbalizer.txt')
    prompt_model = PromptForClassification(plm=plm, template=prompt_template, verbalizer=prompt_verbalizer,
                                           freeze_plm=False).to(device)

    #################################################################################################
    # KPT calibrate  提升了1.1%
    from openprompt.utils.calibrate import calibrate

    support_sampler = FewShotSampler(num_examples_total=200, also_sample_dev=False)
    my_data_set['support'] = support_sampler(my_data_set['train'], seed=1)
    for example in my_data_set['support']:
        example.label = -1  # remove the label s of support set for classification
    support_dataloader = PromptDataLoader(dataset=my_data_set["support"], template=prompt_template, tokenizer=tokenizer,
                                          tokenizer_wrapper_class=wrapper_class, max_seq_length=max_length,
                                          batch_size=4, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                          truncate_method="tail")
    cc_logits = calibrate(prompt_model, support_dataloader)
    prompt_model.verbalizer.register_calibrate_logits(cc_logits)

    #################################################################################################

    torch.cuda.empty_cache()

    # evaluate(valid_dataloader)
    acc = evaluate(test_dataloader, 'test')
    word_nums.append(i)
    acc_nums.append(acc)

plt.figure()
plt.plot(word_nums, acc_nums, label='acc')
plt.title('zero-shot accuracy')
plt.legend()
plt.xlabel('k')
plt.ylabel('acc')

plt.show()