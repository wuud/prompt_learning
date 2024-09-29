
# classes = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
# map = {'体育':0, '财经':1, '房产':2, '家居':3, '教育':4, '科技':5, '时尚':6, '时政':7, '游戏':8, '娱乐':9}

k = 280

def trans_list_to_string(list):
    res = ''
    for item in list:
        res += str(item)
        res += ','
    return res[:-1]

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

