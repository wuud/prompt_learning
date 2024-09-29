# coding=utf-8
import numpy as np
import scipy
from openprompt.plms import load_plm
from openprompt.prompts import MixedTemplate, ManualVerbalizer, ManualTemplate, ProtoVerbalizer
from openprompt import PromptDataLoader
from openprompt import PromptForClassification
from datasets import load_dataset, load_from_disk
from openprompt.data_utils import InputExample
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, BertForMaskedLM, BertConfig
import torch
import matplotlib.pyplot as plt
from openprompt.data_utils.data_sampler import FewShotSampler
from torch.nn import functional as F
from transformers import BertTokenizer, BertModel
from openprompt.utils.reproduciblity import set_seed

torch.autograd.set_detect_anomaly(True)
model_path = 'E://models//chinese-roberta-wwm-ext'
save_path = "./best_val.ckpt"
batch_size = 4
EPOCH = 20
max_length = 512
device = 'cuda'

# zero shot: 0.7644
learning_rate = 1e-6

seed = 143
shot = 1
# 定义数据集
train_file = 'E:\datasets\cnews\\cnews.train.txt'
val_file = 'E:\datasets\\cnews\\cnews.val.txt'
test_file = 'E:\datasets\\cnews\\cnews.test.txt'
plm, tokenizer, model_config, wrapper_class = load_plm("bert", model_path)  # 本地路径

class NeuralNetwork(nn.Module):
    def __init__(self,model_path,output_way):
        super(NeuralNetwork, self).__init__()
        model_config = BertConfig.from_pretrained(model_path)
        self.bert = BertModel.from_pretrained(model_path, config=model_config)

        # self.bert = plm
        self.output_way = output_way
    def forward(self, input_ids, attention_mask, token_type_ids):
        x1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.output_way == 'cls':
            output = x1.last_hidden_state[:,0]
        elif self.output_way == 'pooler':
            output = x1.pooler_output
        return output
class TrainDataset(Dataset):
    def __init__(self, data, tokenizer, maxlen, transform=None, target_transform=None):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.transform = transform
        self.target_transform = target_transform

    def text_to_id(self, source):
        sample = self.tokenizer([source, source], max_length=self.maxlen, truncation=True, padding='max_length',
                                return_tensors='pt')
        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.text_to_id(self.data[idx])


def get_cl_data(file):
    res = []
    with open(file, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            # text = line[3:]
            eles = line.split(',')
            res.append(eles[1])
    return res


training_data = TrainDataset(get_cl_data('./dataset/1.txt'), tokenizer, max_length)
train_dataloader = DataLoader(training_data, batch_size=4)


def compute_loss(y_pred, lamda=0.05):
    idxs = torch.arange(0, y_pred.shape[0], device='cuda')
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    # torch自带的快速计算相似度矩阵的方法
    similarities = similarities - torch.eye(y_pred.shape[0], device='cuda') * 1e12
    # 屏蔽对角矩阵即自身相等的loss
    similarities = similarities / lamda
    # 论文中除以 temperature 超参 0.05
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def train(dataloader, model, optimizer):
    model.train()
    size = len(dataloader.dataset)
    max_corrcoef = 0
    for batch, data in enumerate(dataloader):
        input_ids = data['input_ids'].view(len(data['input_ids']) * 2, -1).to(device)
        attention_mask = data['attention_mask'].view(len(data['attention_mask']) * 2, -1).to(device)
        token_type_ids = data['token_type_ids'].view(len(data['token_type_ids']) * 2, -1).to(device)
        pred = model(input_ids, attention_mask, token_type_ids)
        loss = compute_loss(pred)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            loss, current = loss.item(), batch * int(len(input_ids) / 2)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            model.eval()
            # corrcoef = test(testdata, model)
            model.train()
            # print(f"corrcoef_test: {corrcoef:>4f}")
            # if corrcoef > max_corrcoef:
            #     max_corrcoef = corrcoef
            torch.save(model.state_dict(), save_path)
            # print(f"Higher corrcoef: {(max_corrcoef):>4f}%, Saved PyTorch Model State to model.pth")

model = NeuralNetwork(model_path,'pooler').to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
for t in range(1):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, optimizer)

bert_model_state_dict = torch.load(save_path)
bert_mlm_model_state_dict = plm.state_dict()
print(model.state_dict().keys())
print(plm.state_dict().keys())
# 将 BertModel 的参数加载到 BertForMaskedLM 中
updated_state_dict = {k: v for k, v in bert_model_state_dict.items() if k in bert_mlm_model_state_dict}
print(updated_state_dict.keys())
bert_mlm_model_state_dict.update(updated_state_dict)
plm.load_state_dict(bert_mlm_model_state_dict)


map = {'体育': 0, '财经': 1, '房产': 2, '家居': 3, '教育': 4, '科技': 5, '时尚': 6, '时政': 7, '游戏': 8, '娱乐': 9}
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

sampler = FewShotSampler(num_examples_per_label=shot, also_sample_dev=True, num_examples_per_label_dev=shot)
train_dataset, valid_dataset = sampler(my_data_set['train'], seed=seed)
# print(train_dataset)
# template

# freq > 3, and top40(下同): 0.5328358208955224
# prompt_template = ManualTemplate(text='{"placeholder":"text_a" }。好好思考一下，这包括{"mask"}。', tokenizer=tokenizer) # template1
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

classes = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
label_words = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

prompt_verbalizer = ManualVerbalizer(classes=classes, tokenizer=tokenizer).from_file('./verbalizer.txt')
# prompt_verbalizer = ManualVerbalizer(classes=classes, tokenizer=tokenizer, label_words.txt=label_words.txt)
prompt_model = PromptForClassification(plm=plm, template=prompt_template, verbalizer=prompt_verbalizer,
                                       freeze_plm=False).to(device)

val_accs = []


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
    if type == 'validation':
        val_accs.append(acc)
    print(type, ' accuracy: ', acc)
    return acc


# train

loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)


def train(EPOCH):
    # 这三个list用来绘图
    nums = []
    loss_nums = []
    acc_nums = []
    best_val_acc = 0
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
        val_acc = evaluate(valid_dataloader)
        print("Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)
        nums.append(epoch)
        loss_nums.append(tot_loss / (step + 1))
        if val_acc >= best_val_acc:
            torch.save(prompt_model.state_dict(), save_path)
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


train(EPOCH)
prompt_model.load_state_dict(torch.load(save_path))
prompt_model = prompt_model.cuda()

# evaluate(valid_dataloader)
test_acc = evaluate(test_dataloader, 'test')
# with open(f'./res/seed{seed}_shot{shot}.txt', 'w') as f:
#     f.write('seed = {}, shot = {}, acc = {}'.format(seed, shot, test_acc))
