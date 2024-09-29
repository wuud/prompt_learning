import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertPooler, BertModel
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self, model, model_config, drop_out=0.5):
        super(NeuralNetwork, self).__init__()
        # model_path = 'E://models//chinese-roberta-wwm-ext'
        self.bert = model
        self.bert.pooler = BertPooler(model_config)
        model_config.attention_probs_dropout_prob = drop_out
        model_config.hidden_dropout_prob = drop_out


    def forward(self, input_ids, attention_mask, token_type_ids):
        x1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        print(x1.last_hidden_state.shape)
        output = x1.pooler_output
        print(output.shape)
        return output


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, maxlen):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return len(self.dataset)

    def encode_text(self, text):
        # 使用BERT分词器对文本进行处理和编码，返回输入张量
        encoding = self.tokenizer(text, max_length=self.maxlen, padding="max_length", truncation=True,
                                  return_tensors="pt")
        encoding["input_ids"] = encoding["input_ids"].squeeze()
        encoding["attention_mask"] = encoding["attention_mask"].squeeze()
        encoding["token_type_ids"] = encoding["token_type_ids"].squeeze()
        return encoding

    def __getitem__(self, index):
        anchor_text, positive_text, negative_text = self.dataset[index]

        # 对文本进行处理和编码，获取输入张量
        anchor_input = self.encode_text(anchor_text)
        positive_input = self.encode_text(positive_text)
        negative_input = self.encode_text(negative_text)

        return anchor_input, positive_input, negative_input


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=100.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        # print('distance is : ',distance_positive - distance_negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

# 对锚点和正例 组合 生成更多数据
def gen_data1(pos, neg):
    tripe_list = []
    for i in range(len(pos) - 1):
        for j in range(i + 1, len(pos), 1):
            anchor = pos[i]
            positive = pos[j]
            negative = neg[i]
            tripe_list.append((anchor, positive, negative))

    for i in range(len(neg) - 1):
        for j in range(i + 1, len(neg), 1):
            anchor = neg[i]
            positive = neg[j]
            negative = pos[i]
            tripe_list.append((anchor, positive, negative))
    return tripe_list


# 对正例负例 组合 生成更多数据
def gen_data2(pos, neg):
    tripe_list = []
    for i in range(len(pos)):
        for j in range(len(neg)):
            positive = pos[i]
            negative = neg[j]
            tripe_list.append((positive, positive, negative))  # 锚点和正例都是积极类别样本

            tripe_list.append((negative, negative, positive))
            # print((negative, negative, positive))
    return tripe_list


def get_loader_model(model, config, train_dataset, valid_dataset, tokenizer, max_length, prompt_dataloader_len):
    tripe_list = []
    pos = []
    neg = []
    for i in range(len(train_dataset)):
        if train_dataset[i].label == 0:
            neg.append(train_dataset[i].text_a)
        else:
            pos.append(train_dataset[i].text_a)

        # if valid_dataset[i].label == 0:
        #     neg.append(valid_dataset[i].text_a)
        # else:
        #     pos.append(valid_dataset[i].text_a)

    # print(pos)
    # print(neg)

    # tripe_list = gen_data1(pos, neg) # 8791
    tripe_list = gen_data2(pos, neg)  # 8775

    print('tripe list length: ', len(tripe_list))
    triplet_dataset = TripletDataset(tripe_list, tokenizer, max_length)
    batch_size = min(int(len(tripe_list) / prompt_dataloader_len), 8)
    print('triple batch size: ', batch_size)
    tripe_loder = DataLoader(triplet_dataset, batch_size=batch_size, shuffle=True)
    model = NeuralNetwork(model.bert, config).to('cuda')
    return tripe_loder, model
