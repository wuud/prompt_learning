from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualVerbalizer, ManualTemplate
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertModel
import torch
from transformers.models.bert.modeling_bert import BertPooler

device = 'cuda'

class NeuralNetwork(nn.Module):
    def __init__(self, model, output_way, config, drop_out=0.3):
        super(NeuralNetwork, self).__init__()
        # model_config = BertConfig.from_pretrained(model_path)
        # self.bert = BertModel.from_pretrained(model_path, config=model_config)
        # model_config.attention_probs_dropout_prob = drop_out
        # model_config.hidden_dropout_prob = drop_out
        self.bert = model
        self.bert.pooler = BertPooler(config)
        # config.attention_probs_dropout_prob = drop_out
        # config.hidden_dropout_prob = drop_out

        # self.bert = plm
        self.output_way = output_way

    def forward(self, input_ids, attention_mask, token_type_ids):
        x1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.output_way == 'cls':
            output = x1.last_hidden_state[:, 0]
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


# def get_cl_data(file):
#     res = []
#     with open(file, encoding='utf-8') as f:
#         for idx, line in enumerate(f):
#             # text = line[3:]
#             eles = line.split(',')
#             res.append(eles[1])
#     return res


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

def get_cl_module(train_dataset, tokenizer, max_length, model, config, batch_size=6):
    cl_training_data = TrainDataset(train_dataset, tokenizer, max_length)
    cl_train_dataloader = DataLoader(cl_training_data, batch_size=batch_size)
    # print("dataloader length: ", len(cl_train_dataloader))
    model = NeuralNetwork(model.bert, 'pooler', config).to(device)
    # cl_optimizer = torch.optim.AdamW(model.parameters(), lr=CL_lr)
    return cl_train_dataloader, model


