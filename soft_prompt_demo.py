import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import BertTokenizer, BertModel, AdamW
import matplotlib.pyplot as plt
import matplotlib

max_length = 100 # 句子最大长度
batch_size = 48
eopch = 5
n_tokens = 5
device = torch.device("cuda:0")
initialize_from_vocab = True
model_path = 'E://models/bert-base-chinese'

class SoftEmbedding(nn.Module):
    def __init__(self,
                 wte: nn.Embedding,
                 n_tokens: int = 10,
                 random_range: float = 0.5,
                 initialize_from_vocab: bool = True):
        """appends learned embedding to
        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                                  n_tokens,
                                                                                  random_range,
                                                                                  initialize_from_vocab))

    def initialize_embedding(self,
                             wte: nn.Embedding,
                             n_tokens: int = 10,
                             random_range: float = 0.5,
                             initialize_from_vocab: bool = True):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)

    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        # print('forward tokens : ',tokens.shape)
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        # print("input_embedding: ", input_embedding.shape)
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        # print("learned_embedding: ", learned_embedding.shape)
        return torch.cat([learned_embedding, input_embedding], 1)

#定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 拼接prompt
        input_ids = torch.cat([torch.full((batch_size, n_tokens), 50256).to(device), input_ids[:, :max_length - n_tokens]], 1)
        attention_mask = torch.cat([torch.full((batch_size, n_tokens), 1).to(device), attention_mask[:, :max_length - n_tokens]], 1)

        out = pretrained(input_ids=input_ids,
                   attention_mask=attention_mask,
                   token_type_ids=token_type_ids)

        out = self.fc(out.last_hidden_state[:, 0])

        out = out.softmax(dim=1)

        return out



print('加载预训练模型')
tokenizer = BertTokenizer.from_pretrained(model_path)
pretrained = BertModel.from_pretrained(model_path).to(device)

# 不训练,不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)

s_wte = SoftEmbedding(pretrained.get_input_embeddings(),
                      n_tokens=n_tokens,
                      initialize_from_vocab=initialize_from_vocab)
print('加载soft prompt 层')
pretrained.set_input_embeddings(s_wte)

# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset = load_from_disk('E:\workspace\python\Huggingface_Toturials-main/data/ChnSentiCorp')
        self.dataset = self.dataset[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']

        return text, label

# 数据处理
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    # 编码
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                       truncation=True,
                                       padding='max_length',
                                       max_length=max_length,
                                       return_tensors='pt',
                                       return_length=True)
    # print('original data : ', data)
    # input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    # print(data['length'], data['length'].max())

    return input_ids, attention_mask, token_type_ids, labels


dataset = Dataset('train')
# 数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)

# for i, (input_ids, attention_mask, token_type_ids,
#         labels) in enumerate(loader):
#     break
#
# print(model(input_ids=input_ids,
#       attention_mask=attention_mask,
#       token_type_ids=token_type_ids).last_hidden_state.shape)
# print(labels.shape)

model = Model().to(device)
optim = AdamW(model.parameters(), lr=1e-6)
loss_func = torch.nn.CrossEntropyLoss()

model.train()

# 这三个list用来绘图
nums = []
loss_nums = []
acc_nums = []

for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
    print('nums = ', i)
    input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), labels.to(device)
    out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    # print(out.shape)
    # print(labels.shape)
    loss = loss_func(out, labels)
    loss.backward()
    optim.step()
    optim.zero_grad()

    if i % 5 == 0:
        out = out.argmax(dim=1)
        acc = (out == labels).sum().item() / len(labels)
        print('i = {}, loss = {}, acc = {}'.format(i, loss, acc))
        nums.append(i)
        loss_nums.append(loss.item())
        acc_nums.append(acc)

    # if i == 25:
    #     break
    # break


plt.figure()
plt.plot(nums, acc_nums, label='Train acc')
# plt.plot(epoch, eval_acc, label='Test acc')
plt.title('Training and Testing accuracy')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.figure()
plt.plot(nums, loss_nums, label='Train loss')
# plt.plot(epoch, eval_loss, label='Test loss')
plt.title('Training and Testing loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')

plt.show()