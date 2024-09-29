import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
import codecs
import re
'''
80.6±1.7(81.9) 83.1±2.4(85.7) 74.2±2.5(77.2)
79.9±5.9(84.6) 82.1±2.9(85.5) 75.2±8.5(85.5)
84.9±2.6(86.8) 84.3±1.5(86.0) 73.6±0.9(74.3)
82.8±1.5(84.1) 85.4±0.9(86.5) 78.3±1.1(79.1)
81.3±4.2(84.5) 85.2±1.0(86.3) 76.2±1.6(77.6)
84.6±2.8(86.7) 85.4±0.7(86.2) 78.8±0.7(79.6)
84.8±0.2(85.0) 85.4±0.5(86.0) 79.3±1.2(80.6)
86.1±1.0(87.3) 85.3±0.1(85.4) 77.6±0.9(78.2)
85.8±1.2(87.2) 86.3±0.1(86.5) 79.6±1.2(80.7)
85.4±0.1(85.5) 85.7±1.1(86.5) 80.0±1.3(81.0)
86.3±0.9(87.1) 85.8±1.0(86.5) 77.4±1.1(78.1)
86.4±0.2(86.7) 86.2±2.2(87.2) 80.2±0.5(80.6)
'''

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

for i in range(12):
    method = i % 3
    if method == 0:
        FT.append(datas[i])
    elif method == 1:
        PT.append(datas[i])
    elif method == 2:
        SOFT.append(datas[i])
    # elif method == 3:
    #     Our.append(datas[i])
    # elif method == 4:
    #     AT.append(datas[i])

for i in range(12):
    method = i % 3
    if method == 0:
        FT_e.append(errs[i])
    elif method == 1:
        PT_e.append(errs[i])
    elif method == 2:
        SOFT_e.append(errs[i])
    # elif method == 3:
    #     Our_e.append(errs[i])
    # elif method == 4:
    #     AT_e.append(errs[i])

FT = list(map(list, zip(*FT)))
FT = [[float(num) for num in sublist] for sublist in FT]
FT_e = list(map(list, zip(*FT_e)))
FT_e = [[float(num) for num in sublist] for sublist in FT_e]
PT = list(map(list, zip(*PT)))
PT = [[float(num) for num in sublist] for sublist in PT]
PT_e = list(map(list, zip(*PT_e)))
PT_e = [[float(num) for num in sublist] for sublist in PT_e]
SOFT = list(map(list, zip(*SOFT)))
SOFT = [[float(num) for num in sublist] for sublist in SOFT]
SOFT_e = list(map(list, zip(*SOFT_e)))
SOFT_e = [[float(num) for num in sublist] for sublist in SOFT_e]
# Our = list(map(list, zip(*Our)))
# Our = [[float(num) for num in sublist] for sublist in Our]
# Our_e = list(map(list, zip(*Our_e)))
# Our_e = [[float(num) for num in sublist] for sublist in Our_e]
# AT = list(map(list, zip(*AT)))
# AT = [[float(num) for num in sublist] for sublist in AT]
# AT_e = list(map(list, zip(*AT_e)))
# AT_e = [[float(num) for num in sublist] for sublist in AT_e]



labels = [1, 4, 8, 16]
x= [1,2,3,4]

# 创建画布和子图
plt.xticks(x, labels)
# fig, axs = plt.subplots(2, 2)
fig = plt.figure(figsize=(10, 12))  # 设置整个图的大小
gs = gridspec.GridSpec(2, 15, height_ratios=[1,1])  # 设置网格布局和每行的高度比例

# 绘制第一个小图
ax1 = fig.add_subplot(gs[0, :7])
ax1.plot(x, FT[0], label='PL', marker='o')
ax1.plot(x, PT[0], label='PL+CL', marker='s')

ax1.plot(x, SOFT[0], label='PL+ML', marker='^')

# ax1.plot(x, Our[0], label='Our', marker='h')
# ax1.fill_between(x, np.subtract(Our[0], Our_e[0]), np.add(Our[0], Our_e[0]), alpha=0.3)
#
# ax1.plot(x, AT[0], label='Our', marker='h')
# ax1.fill_between(x, np.subtract(AT[0], AT_e[0]), np.add(AT[0], AT_e[0]), alpha=0.3)

ax1.set_title('ChnSentiCorp')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_xlabel('Shot')
ax1.set_ylabel('Accuracy')
# ax1.set_yticks([20, 25, 30, 35, 40, 45, 50, 55])  # 设置纵坐标刻度位置
# ax1.set_yticklabels(['20', '25', '30', '35', '40', '45', '50', '55'])  # 设置纵坐标刻度标签
ax1.legend()

# 绘制第二个小图
ax2 = fig.add_subplot(gs[0, 8:])
ax2.plot(x, FT[1], label='PL', marker='o')
ax2.plot(x, PT[1], label='PL+CL', marker='s')

ax2.plot(x, SOFT[1], label='PL+ML', marker='^')
#
ax2.set_title('EPRSTMT')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_xlabel('Shot')
ax2.set_ylabel('Accuracy')
ax2.legend()
#
# 绘制第三个小图
ax3 = fig.add_subplot(gs[1, 4:11])
ax3.plot(x, FT[2], label='PL', marker='o')
ax3.plot(x, PT[2], label='PL+CL', marker='s')

ax3.plot(x, SOFT[2], label='PL+ML', marker='^')

ax3.set_title('Hotel')
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.set_xlabel('Shot')
ax3.set_ylabel('Accuracy')
ax3.legend()

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
fig.suptitle("度量学习对比实验", fontsize='18')

dpi = 300
fig.set_dpi(dpi)

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()