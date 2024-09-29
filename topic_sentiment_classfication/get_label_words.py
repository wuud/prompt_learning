"""
1. 对数据集进行分词
2. 拿到所有形容词
3. 拿到正向情绪数据集内的所有正向形容词，拿到负向情绪数据集内的所有负向形容词
    3.1
"""

import jieba
import jieba.posseg as pseg
from sentence_transformers import SentenceTransformer, util
import codecs

model_path = 'E://models//paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(model_path).to('cuda')


def get_hotel_data():
    res = []
    train_data_path = "./dataset/hotel_train.txt"
    lines = codecs.open(train_data_path, encoding='utf-8').readlines()
    for idx, line in enumerate(lines):
        eles = line.split(',')
        res.append(eles[0])
    return res


def filter(word, embedding1, embedding2):
    embedding3 = model.encode(word, convert_to_tensor=True)
    score1 = util.pytorch_cos_sim(embedding1, embedding3)[0][0].item()
    score2 = util.pytorch_cos_sim(embedding2, embedding3)[0][0].item()
    # print('score: ', score1, score2)
    if (score1 < 0.65 and score2 < 0.65) or abs(score1 - score2) < 0.015:
        return -1
    elif score1 > score2:
        return 1
    else:
        return 2


def get_words_from_dataset():
    good = '好'
    bad = '差'
    embedding1 = model.encode(good, convert_to_tensor=True)
    embedding2 = model.encode(bad, convert_to_tensor=True)

    data = get_hotel_data()
    pos = {}
    neg = {}
    for sentence in data:
        # 使用结巴分词进行分词
        # words = pseg.cut(sentence)
        # 遍历分词结果，提取形容词

        prev_word, prev_flag = None, None
        for word, flag in pseg.cut(sentence):
            if flag == 'a':  # 形容词标记 'a'

                score = filter(word, embedding1, embedding2)
                # print(word, score)
                if score < 0: continue
                if score == 1:
                    pos[word] = pos.get(word, 0) + 1
                else:
                    neg[word] = neg.get(word, 0) + 1
            # prev_word, prev_flag = word, flag
            # print('word: {}, flag: {}'.format(word, flag))

    pos = sorted(pos.items(), key=lambda x: x[1], reverse=True)
    neg = sorted(neg.items(), key=lambda x: x[1], reverse=True)
    print('=' * 100)
    print(pos)
    print(neg)
    threshold = 3
    pos_res = [item[0] for item in pos if item[1] >= threshold]
    neg_res = [item[0] for item in neg if item[1] >= threshold]

    print(pos_res)
    print(neg_res)


if __name__ == '__main__':
    get_words_from_dataset()
