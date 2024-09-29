import json

l1 = 0 # < 50
l2 = 0 #  50 - 100
l3 = 0 # 100 - 256
l4 = 0 # 256 - 512
l5 = 0 # 512 - 1024
l6 = 0 # > 1024
l7 = 0 # > 1024


# 643 10650 38574 3491 1 1
my_data_set = {}
train_file = 'E:\datasets\\tnews\\train.json'

with open(train_file, encoding='utf-8') as f:
    for idx, line in enumerate(f):
        line = json.loads(line)
        text_len = len(line['sentence'])
        if text_len < 16:
            l1 += 1
        elif text_len < 32:
            l2 += 1
        elif text_len < 64:
            l3 += 1
        elif text_len < 128:
            l4 += 1
        elif text_len < 256:
            l5 += 1
        elif text_len < 512:
            l6 += 1
        else:
            l7 += 1

print(l1, l2, l3, l4, l5, l6, l7)

