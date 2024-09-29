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
train_file = 'E:\datasets\cnews\\cnews.train.txt'
val_file = 'E:\datasets\\cnews\\cnews.val.txt'
test_file = 'E:\datasets\\cnews\\cnews.test.txt'
map = {'体育':0, '财经':1, '房产':2, '家居':3, '教育':4, '科技':5, '时尚':6, '时政':7, '游戏':8, '娱乐':9}
my_data_set = {}
def get_data_set(file, split):
    my_data_set[split] = []
    with open(file, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            label = line[:2]
            text = line[3:]
            input = InputExample(guid=idx, label=map[label], text_a=text)
            my_data_set[split].append(input)
get_data_set(train_file, 'train')
# get_data_set(val_file, 'validation')
get_data_set(test_file, 'test')

print('train dataset length: ', len(my_data_set['train']))
# print('validation dataset length: ', len(my_data_set['validation']))
print('test dataset length: ', len(my_data_set['test']))

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
    with open('./verbalizer_100.txt', 'r', encoding='gbk') as f:
        for line in f:
            words = line.split(',')
            # print(len(words))
            if len(words) >= k:
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

classes = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

acc_nums = []
word_nums = []
for i in range(200, 510, 10):
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