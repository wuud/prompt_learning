import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
import codecs
import re


file = '1.txt'

datas = []
errs = []


lines = codecs.open(file, encoding='utf-8').readlines()
for line in lines:
    nums = re.split(r'\s+', line)
    ds = []
    es = []
    for num in nums:
        if len(num) == 0:
            continue
        main_value = num.split('±')[0]

        # 使用切片操作获取误差部分
        error = num.split('±')[1].split('(')[0]
        # print(main_value, error)
        ds.append(main_value)
        es.append(error)
    # print(ds, es)
    assert len(ds) == 3
    assert len(es) == 3
    datas.append(ds)
    errs.append(es)

# print(datas)
# print(errs)

FT =[]
FT_e = []
PT  = []
PT_e = []
SOFT  = []
SOFT_e = []
Our  = []
Our_e = []
AT  = []
AT_e = []

for i in range(8):
    method = i % 2
    if method == 0:
        FT.append(datas[i])
    elif method == 1:
        PT.append(datas[i])


for i in range(8):
    method = i % 2
    if method == 0:
        FT_e.append(errs[i])
    elif method == 1:
        PT_e.append(errs[i])

FT = list(map(list, zip(*FT)))
FT = [[float(num) for num in sublist] for sublist in FT]
FT_e = list(map(list, zip(*FT_e)))
FT_e = [[float(num) for num in sublist] for sublist in FT_e]
PT = list(map(list, zip(*PT)))
PT = [[float(num) for num in sublist] for sublist in PT]
PT_e = list(map(list, zip(*PT_e)))
PT_e = [[float(num) for num in sublist] for sublist in PT_e]



labels = [1, 4, 8, 16]
x= [1,2,3,4]

# 创建画布和子图
plt.xticks(x, labels)
# fig, axs = plt.subplots(2, 2)
fig = plt.figure(figsize=(10, 12))  # 设置整个图的大小
gs = gridspec.GridSpec(2, 11, height_ratios=[1,1])  # 设置网格布局和每行的高度比例

# 绘制第一个小图
ax1 = fig.add_subplot(gs[0, :5])
ax1.plot(x, FT[0], label='PL+ML', marker='o')
ax1.plot(x, PT[0], label='PL+ML+AT', marker='s')


ax1.set_title('ChnSentiCorp')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_xlabel('Shot')
ax1.set_ylabel('Accuracy')

ax1.legend()

# 绘制第二个小图
ax2 = fig.add_subplot(gs[0, 6:])
ax2.plot(x, FT[1], label='PL+ML', marker='o')
ax2.plot(x, PT[1], label='PL+ML+AT', marker='s')

#
ax2.set_title('EPRSTMT')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_xlabel('Shot')
ax2.set_ylabel('Accuracy')
ax2.legend()
#
# 绘制第三个小图
ax3 = fig.add_subplot(gs[1, 3:8])
ax3.plot(x, FT[2], label='PL+ML', marker='o')
ax3.plot(x, PT[2], label='PL+ML+AT', marker='s')


ax3.set_title('Hotel')
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.set_xlabel('Shot')
ax3.set_ylabel('Accuracy')
ax3.legend()

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
fig.suptitle("对抗训练对比实验", fontsize='18')

dpi = 300
fig.set_dpi(dpi)

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()