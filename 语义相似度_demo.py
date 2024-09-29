from sentence_transformers import SentenceTransformer, util
import torch

# model_path = 'E://models//ChineseBERT-base'
# model_path = 'E://models//chinese-roberta-wwm-ext'
model_path = 'E:/models/paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(model_path).to('cuda')
# model= SentenceTransformer('path-to-your-pretrained-model/paraphrase-MiniLM-L12-v2/')
# 如果是想加载本地的预训练模型，则类似于huggingface的from_pretrained方法，把输入参数换成本地模型的路径
# 计算编码
good = '好'
bad = '差'
word = '一般'
embedding1 = model.encode(good, convert_to_tensor=True)
print('shape', embedding1.shape)
embedding2 = model.encode(bad, convert_to_tensor=True)
embedding3 = model.encode(word, convert_to_tensor=True)
# # 计算语义相似度
cosine_score1_3 = util.pytorch_cos_sim(embedding1, embedding3)
cosine_score2_3 = util.pytorch_cos_sim(embedding3, embedding2)

# cosine_score5_4 = util.pytorch_cos_sim(embedding5, embedding4)
print(good, word, "语义相似度是：", cosine_score1_3[0][0])
print(bad, word, "语义相似度是：", cosine_score2_3)

print((cosine_score2_3[0][0] - cosine_score1_3[0][0]).item())
#
# a = torch.cosine_similarity(embedding1, embedding2, 0)
# print(a)

# print(embedding2 == embedding3)
# print(sentence1,word4,"语义相似度是：",cosine_score1_4)
# print(word5,word4,"语义相似度是：",cosine_score5_4)

# query_embedding = embedding1
# corpus = [embedding1, embedding3, embedding2, embedding4, embedding5]
# # print(embedding3.shape)
# # print(corpus.shape)
# hits = util.semantic_search(query_embedding, corpus_embeddings=corpus, top_k=5)
# word_indexs = []
# for word_index in hits[0]:
#     word_indexs.append(word_index['corpus_id'])
# print(hits[0])
# print(word_indexs)
