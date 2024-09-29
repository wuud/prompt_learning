
def trans_list_to_string(list):
    res = ''
    for item in list:
        res += str(item)
        res += ','
    return res[:-1]


def process_labels(k):
    res = ''
    with open('./verbalizer_300.txt', 'r', encoding='gbk') as f:
        for line in f:
            words = line.split(',')
            # print(len(words))
            if len(words) > k:
                res += trans_list_to_string(words[:k])
                res += '\n'
            else:
                res += trans_list_to_string(words)
    print(res)
    with open('./verbalizer.txt', 'w', encoding='gbk') as f:
        f.write(res)

process_labels(40)