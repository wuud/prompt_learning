import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BertTokenizer, BertModel



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
        print('forward')
        print(tokens)
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        print("input_embedding: ", input_embedding.shape)
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        print("learned_embedding: ", learned_embedding.shape)
        return torch.cat([learned_embedding, input_embedding], 1)


n_tokens = 5
initialize_from_vocab = True
model_path = 'E:/models/gpt2'
print('加载预训练模型GPT2')
tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
s_wte = SoftEmbedding(model.get_input_embeddings(),
                      n_tokens=n_tokens,
                      initialize_from_vocab=initialize_from_vocab)
print('加载soft prompt 层')
model.set_input_embeddings(s_wte)

inputs = tokenizer("May the force be", return_tensors="pt")
# print(inputs['input_ids'].shape)
print('----------------------')
# need to pad attention_mask and input_ids to be full seq_len + n_learned_tokens
# even though it does not matter what you pad input_ids with, it's just to make HF happy
inputs['input_ids'] = torch.cat([torch.full((1, n_tokens), 50256), inputs['input_ids']], 1)
inputs['attention_mask'] = torch.cat([torch.full((1, n_tokens), 1), inputs['attention_mask']], 1)
print('1111111111111')
outputs = model(**inputs)
print(inputs['input_ids'].shape)
print(inputs)
print(tokenizer.decode(inputs['input_ids'][0]))
# print(outputs)

bert_path = 'E:/models/bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
bert_model = BertModel.from_pretrained(bert_path)
inputs = bert_tokenizer("May the force be", return_tensors="pt")
print(inputs,'===============')