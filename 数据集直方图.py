import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# 每个数据集在不同长度范围内的样本数量
# dataset1_counts = [20, 35, 45, 30, 10, 5, 2]  # 示例数据，可以替换为实际数据
# dataset2_counts = [15, 25, 40, 20, 8, 4, 1]  # 示例数据，可以替换为实际数据
# dataset3_counts = [10, 20, 30, 25, 12, 6, 3]  # 示例数据，可以替换为实际数据
tnews = [11293, 38574, 3491, 1, 1, 0, 0]
cnews = [2, 161, 662, 3124, 4851, 9835, 31365]
csldcp = [1, 2, 64, 430, 1030, 466, 43]
# 计算每个数据集的总样本数
dataset1_total = sum(tnews)
dataset2_total = sum(cnews)
dataset3_total = sum(csldcp)

# 计算每个数据集在范围内的样本数量占总样本数的百分比
dataset1_percentages = [count / dataset1_total * 100 for count in tnews]
dataset2_percentages = [count / dataset2_total * 100 for count in cnews]
dataset3_percentages = [count / dataset3_total * 100 for count in csldcp]

# 定义长度范围和范围标签
bins = ['0-16', '16-32', '32-64', '64-128', '128-256', '256-512', '512及以上']

# 绘制直方图
fig, ax = plt.subplots(dpi=300)
x = np.arange(len(bins))
width = 0.3

rects1 = ax.bar(x - width, dataset1_percentages, width, label='Tnews')
rects2 = ax.bar(x, dataset2_percentages, width, label='THUCNews')
rects3 = ax.bar(x + width, dataset3_percentages, width, label='CSLDCP')

# 设置坐标轴和标题
ax.set_xlabel('长度范围')
ax.set_ylabel('样本数量百分比')
ax.set_title('不同数据集样本长度分布直方图')
ax.set_xticks(x)
ax.set_xticklabels(bins)
ax.legend()

# 添加数据标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 1:
            ax.annotate('{:.1f}%'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',fontsize=8)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# # 正确显示中文和负号
# plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = ['NSimSun']
plt.rcParams["axes.unicode_minus"] = False
# 显示图形
plt.show()