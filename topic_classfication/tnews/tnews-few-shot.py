# coding=utf-8

import json

from openprompt.plms import load_plm
from openprompt.prompts import MixedTemplate, ManualVerbalizer, ManualTemplate, ProtoVerbalizer
from openprompt import PromptDataLoader
from openprompt import PromptForClassification
from datasets import load_dataset, load_from_disk
from openprompt.data_utils import InputExample
from torch.optim import AdamW
import torch
import matplotlib.pyplot as plt
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.utils.reproduciblity import set_seed
import random
torch.autograd.set_detect_anomaly(True)
# model_path = 'E://models//wobert_chinese_base'
model_path = 'E://models//chinese-roberta-wwm-ext'
batch_size = 4
EPOCH = 20
max_length = 50
device = 'cuda'
# 2,144,145
# seed = 2, 1: 0.5427860696517413, 4: 0.5502487562189055, 8：0.5671641791044776, 16: 0.5781094527363184 , 32: 0.5781094527363184  , 64 : 0.5761194029850746
# seed = 144, 1: 0.5577114427860697, 4: 0.5606965174129354, 8:0.572139303482587, 16: 0.5691542288557214 , 32: 0.5691542288557214 , 64: 0.5756218905472636
# seed = 145, 1: 0.56318407960199, 4: 0.5527363184079602,8: 0.5601990049751244, 16: 0.5810945273631841 ,
shot = 16
# learning_rate = 5e-6
learning_rate = 1e-6
# 0.5621890547263682
seed = 2
'''
应考虑，soft token个数对学习率的影响
1. 对不同soft token 进行对比
1.1 soft token的数量
1.2 soft token + hard prompt
2. 对不同 verbalizer 进行对比
2.1 单个verbalizer 和 每个标签多个 verbalizer 进行对比


zero-shot few-shot 对比
'''
all_core_words = [['故事', '小说'], ['文化', '文明', '诗词', '书法', '书画'], ["娱乐", '音乐', '影视', '综艺', '明星'],
                  ["体育", '运动', '篮球', '足球', '排球'], ["财经", '金融'], ["房产", '房地产'], ["汽车", '轿车'], ["教育", '教书'], ["科技", '技术'],
                  ["军事", '军队'], ["旅游", '旅行'], ["国际", '世界'], ["证券", '股票'], ["农业", '农学'], ["电竞", '游戏']]


set_seed(seed)
# 定义 PLM
plm, tokenizer, model_config, wrapper_class = load_plm("bert", model_path)  # 本地路径
# plm, tokenizer, model_config, wrapper_class = load_plm("bert", 'bert-base-chinese') # huggingface 仓库

# 定义数据集
train_file = 'E:\datasets\\few-tnews\\train_few_all.json'
# train_file = 'E:\datasets\\tnews\\train.json'
# val_file = 'E:\datasets\\tnews\\dev.json'
val_file = 'E:\datasets\\tnews\\dev_few_all.json'
test_file = 'E:\datasets\\few-tnews\\test_public.json'


def trans_label(label):
    map = {100: 0, 101: 1, 102: 2, 103: 3, 104: 4, 106: 5, 107: 6, 108: 7, 109: 8, 110: 9, 112: 10, 113: 11, 114: 12,
           115: 13, 116: 14}
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
# print('test dataset length: ', len(my_data_set['test']))

sampler = FewShotSampler(num_examples_per_label=shot, also_sample_dev=True, num_examples_per_label_dev=shot)
train_dataset, valid_dataset = sampler(my_data_set['train'],seed=seed)
# print(train_dataset)
# template

# prompt_template = ManualTemplate(text='{"placeholder":"text_a" }。好好思考一下，这包括{"mask"}。', tokenizer=tokenizer)  # template1
prompt_template = ManualTemplate(text='{"placeholder":"text_a" }。这包括{"mask"}。', tokenizer=tokenizer)  # template1



train_dataloader = PromptDataLoader(dataset=train_dataset, template=prompt_template, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=wrapper_class, max_seq_length=max_length)
valid_dataloader = PromptDataLoader(dataset=valid_dataset, template=prompt_template, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=wrapper_class, max_seq_length=max_length,
                                    decoder_max_length=3,
                                    batch_size=batch_size, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="head")
test_dataloader = PromptDataLoader(dataset=my_data_set['test'], template=prompt_template, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=wrapper_class, max_seq_length=max_length,
                                   decoder_max_length=3,
                                   batch_size=batch_size, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                   truncate_method="head")

classes = ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游", "国际", "证券", "农业", "电竞"]
label_words = ["民生", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游", "国际", "证券", "农业", "电竞" ]

# 0.5348258706467661
prompt_verbalizer = ManualVerbalizer(classes=classes, tokenizer=tokenizer).from_file('verbalizer.txt')
# prompt_verbalizer = ManualVerbalizer(classes=classes, tokenizer=tokenizer, label_words.txt=label_words.txt)
prompt_model = PromptForClassification(plm=plm, template=prompt_template, verbalizer=prompt_verbalizer,
                                       freeze_plm=False).to(device)

#################################################################################################
# KPT calibrate
from openprompt.utils.calibrate import calibrate

# support_sampler = FewShotSampler(num_examples_total=200, also_sample_dev=False)
support_sampler = FewShotSampler(num_examples_per_label=100, also_sample_dev=False)
my_data_set['support'] = support_sampler(my_data_set['train'], seed=1)
print('support length: ', len(my_data_set['support']))
for example in my_data_set['support']:
    example.label = -1  # remove the labels of support set for classification
support_dataloader = PromptDataLoader(dataset=my_data_set["support"], template=prompt_template, tokenizer=tokenizer,
                                      tokenizer_wrapper_class=wrapper_class, max_seq_length=max_length,
                                      batch_size=8, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                      truncate_method="tail")
cc_logits = calibrate(prompt_model, support_dataloader)
print('cc logits shape: ', cc_logits.shape)
prompt_model.verbalizer.register_calibrate_logits(cc_logits)

#################################################################################################


val_accs = []


def evaluate(model, data_loader, type='validation'):
    all_preds = []
    all_labels = []
    for step, inputs in enumerate(data_loader):
        inputs.cuda()
        logits = model(inputs)
        labels = inputs['label']
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        # print('-' * 100)
        # print('labels is : ', labels.tolist())
        # print('prediction is : ',torch.argmax(logits, dim=-1).tolist())
        # print(logits.shape)
    acc = sum([int(i == j) for i, j in zip(all_preds, all_labels)]) / len(all_preds)
    if type == 'validation':
        val_accs.append(acc)
    print(type, ' accuracy:', acc)
    return acc


# train

loss_func = torch.nn.CrossEntropyLoss()

no_decay = ['bias', 'LayerNorm.weight']

# optimizer_grouped_parameters1 = [
#     {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#     {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
# ]

# Using different optimizer for prompt parameters and model parameters
optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)


# print('++' * 100)
# print(prompt_model.verbalizer.parameters())

def train(EPOCH):
    # 这三个list用来绘图
    nums = []
    loss_nums = []
    acc_nums = []
    best_val_acc  = 0
    print('-------------------start training，EPOCH = {}------------------'.format(EPOCH))
    for epoch in range(EPOCH):  # Longer epochs are needed
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            inputs = inputs.to(device)
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            # torch.nn.utils.clip_grad_norm_(prompt_model.template.parameters(), 1.0)
            # optimizer1.step()
            # optimizer1.zero_grad()
            optimizer.step()
            optimizer.zero_grad()
        val_acc = evaluate(prompt_model, valid_dataloader)
        print("Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)
        nums.append(epoch)
        loss_nums.append(tot_loss / (step + 1))
        if val_acc >= best_val_acc:
            torch.save(prompt_model.state_dict(), "./best_val.ckpt")
            best_val_acc = val_acc

    # 打印一下画图的参数，之后用到的时候可以直接拿来画图
    print(nums)
    print(loss_nums)

    plt.figure()
    plt.plot(nums, loss_nums, label='Train loss')
    plt.plot(nums, val_accs, label='Validation acc')
    plt.title('Training loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.show()

#
# train(EPOCH)
# prompt_model.load_state_dict(torch.load("./best_val.ckpt"))
# prompt_model = prompt_model.cuda()
#
# torch.cuda.empty_cache()

evaluate(prompt_model, test_dataloader, 'test')
