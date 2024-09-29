import matplotlib.pyplot as plt
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import codecs


def reduce_dimensions(embeddings, method='pca', n_components=2):
    # embeddings: 词嵌入表示的向量列表
    # method: 降维方法，可以是 'pca' 或 'tsne'
    # n_components: 降维后的维度

    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components)
    else:
        raise ValueError("Unsupported dimensionality reduction method. Choose either 'pca' or 'tsne'.")

    reduced_embeddings = reducer.fit_transform(embeddings)
    # print(reduced_embeddings)
    return reduced_embeddings


def plot_word_embeddings(cls_embeddings, all_embeddings, labels):
    # embeddings: 词嵌入表示的降维后的向量列表
    # words: 对应于词嵌入的词语列表
    print(labels)
    # 创建一个新的图形
    plt.figure(figsize=(10, 10))

    # 绘制散点图
    for idx, embeddings in enumerate(all_embeddings):
        plt.scatter([emb[0] for emb in embeddings], [emb[1] for emb in embeddings], marker='o', s=30, label=labels[idx])
    for idx, embeddings in enumerate(cls_embeddings):
        plt.scatter(embeddings[0], embeddings[1], marker='p', edgecolors='red', facecolors='none', s=100)

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.legend()
    # 添加词语标签
    # for i, word in enumerate(words):
    #     plt.annotate(word, xy=(embeddings[i][0], embeddings[i][1]), xytext=(5, 2),
    #                  textcoords='offset points', ha='right', va='bottom')

    # 显示图形
    plt.show()

def draw_label_words(model):
    classes = ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游", "国际", "证券", "农业", "电竞"]
    cls_embeddings = []
    for cls in classes:
        emb = model.encode(cls, convert_to_tensor=True)
        cls_embeddings.append(emb.cpu().numpy())

    cls_embeddings = reduce_dimensions(np.array(cls_embeddings))

    file = './tnews/verbalizer.txt'
    lines = codecs.open(file, encoding='utf-8').readlines()
    all_words = []
    for line in lines:
        words = line.split(',')
        all_words.append(words)

    # model_path = 'E:/models/paraphrase-multilingual-MiniLM-L12-v2'

    all_embeddings = []
    for words in all_words:
        embeddings = []
        for word in words:
            emb = model.encode(word, convert_to_tensor=True)
            embeddings.append(emb.cpu().numpy())

        embeddings = reduce_dimensions(np.array(embeddings))
        all_embeddings.append(embeddings)

    # torch.cpu(embeddings)

    # 调用函数绘制散点图
    plot_word_embeddings(cls_embeddings, all_embeddings, classes)

model_path = 'E:/models/chinese-roberta-wwm-ext'
model = SentenceTransformer(model_path)
draw_label_words(model)