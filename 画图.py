import matplotlib.pyplot as plt

x= [1,2,3,4]
labels = [1, 4, 8, 16]
y = [4, 1, 2, 1]
####################################################################
# 散点图

# # https://blog.csdn.net/qq_42804105/article/details/124296152
# #定义颜色变量
# color = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
# plt.scatter(x, y, c=color[2], marker='v')
# plt.show()
#
# ####################################################################
# 折线图
# plt.figure()
#
# plt.xticks(x, labels)
# plt.plot(x, y, label='Train loss', marker='o')
# # plt.plot(y, x, label='Validation acc')
# plt.title('Training loss')
# plt.legend()
# plt.xlabel('epoch')
# plt.ylabel('loss')
#
# plt.show()
#################################多组折线图#####################################
# 创建示例数据
# from matplotlib import gridspec
#
# labels = [1, 4, 8, 16]
#
# x= [1,2,3,4]
# FT1 = [54.6, 81.7, 88, 92]
# PT1 = [63, 74.7, 80.3, 82.2]
# SOFT1 = [53.8, 81.1, 92, 94.1]
# Our1 = [79.6, 86.9, 90.3, 94.2]
#
# FT2 = [25.6, 41.7, 46.9, 50.3]
# PT2 = [40.2, 49.9, 51.3, 54.3]
# SOFT2 = [21.0, 36.7, 45.3, 51.8]
# Our2 = [55.4, 55.4, 56.6, 57.6]
#
# FT3 = [26.1, 42.8, 48.5, 52.8]
# PT3 = [38.2, 44.3, 46.8, 49.4]
# SOFT3 = [17.3, 40.9, 52.0, 52.9]
# Our3 = [48.4, 51.2, 53.3, 56]
#
# # 创建画布和子图
# plt.xticks(x, labels)
# # fig, axs = plt.subplots(2, 2)
# fig = plt.figure(figsize=(10, 12))  # 设置整个图的大小
# gs = gridspec.GridSpec(2, 2, height_ratios=[1,1])  # 设置网格布局和每行的高度比例
#
# # 绘制第一个小图
# ax1 = fig.add_subplot(gs[0, :1])
# ax1.plot(x, FT2, label='FT', marker='o')
# ax1.plot(x, PT2, label='PT', marker='s')
# ax1.plot(x, SOFT2, label='SOFT', marker='^')
# ax1.plot(x, Our2, label='Our', marker='h')
# ax1.set_title('Tnews')
# ax1.set_xticks(x)
# ax1.set_xticklabels(labels)
# ax1.set_xlabel('Shot')
# ax1.set_ylabel('Accuracy')
# ax1.set_yticks([20, 25, 30, 35, 40, 45, 50, 55])  # 设置纵坐标刻度位置
# ax1.set_yticklabels(['20', '25', '30', '35', '40', '45', '50', '55'])  # 设置纵坐标刻度标签
# ax1.legend()
#
# # 绘制第二个小图
# ax2 = fig.add_subplot(gs[0, 1:2])
#
# ax2.plot(x, FT1, label='FT', marker='o')
# ax2.plot(x, PT1, label='PT', marker='s')
# ax2.plot(x, SOFT1, label='SOFT', marker='^')
# ax2.plot(x, Our1, label='Our', marker='h')
# ax2.set_title('THUCNews')
# ax2.set_xticks(x)
# ax2.set_xticklabels(labels)
# ax2.set_xlabel('Shot')
# ax2.set_ylabel('Accuracy')
# ax2.legend()
#
# # 绘制第三个小图
# ax3 = fig.add_subplot(gs[1, :1])
# ax3.plot(x, FT3, label='FT', marker='o')
# ax3.plot(x, PT3, label='PT', marker='s')
# ax3.plot(x, SOFT3, label='SOFT', marker='^')
# ax3.plot(x, Our3, label='Our', marker='h')
# ax3.set_title('CSLDCP')
# ax3.set_xticks(x)
# ax3.set_xticklabels(labels)
# ax3.set_xlabel('Shot')
# ax3.set_ylabel('Accuracy')
# ax3.legend()
#
# # axs[1, 1].axis('off')
#
# plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
# fig.suptitle("不同方法在三个数据集上的表现", fontsize='18')
#
# dpi = 300
# fig.set_dpi(dpi)
#
# # 调整子图之间的间距
# plt.tight_layout()
#
# # 显示图形
# plt.show()
####################################################################

# 准备数据
# x_data = [f"20{i}年" for i in range(16, 19)]
# y_data = [100,200,300]
#
# # 正确显示中文和负号
# plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt.rcParams["axes.unicode_minus"] = False
#
# # 画图，plt.bar()可以画柱状图
# for i in range(len(x_data)):
# 	plt.bar(x_data[i], y_data[i],width=0.5)
# # 设置图片名称
# plt.title("销量分析")
# # 设置x轴标签名
# plt.xlabel("年份")
# # 设置y轴标签名
# plt.ylabel("销量")
# # 显示
# plt.show()


#构造数据
import numpy as np
# y1 = [1,4,6,8,9,4,3,8]
# y2 = [2,5,9,5,3,2,7,4]
# x = [0,1,2,3,4,5,6,7]
#
# #设置柱状图的宽度
# width = 0.4
#
# #绘图
# plt.figure(figsize=(8,4))
#
# plt.bar(x=x,height=y1,width=width,label='Data1')
# plt.bar(x=[item+width for item in x],height=y2,width=width, label='Data2')
#
# #添加数据标签
# # for x_value,y_value in zip(x,y1):
# #     plt.text(x=x_value,y=y_value,s=y_value)
# #
# # for x_value,y_value in zip(x,y2):
# #     plt.text(x=x_value+width,y=y_value,s=y_value)
#
# #添加图标题和图例
# plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt.rcParams["axes.unicode_minus"] = False
# plt.title('并列柱状图')
# plt.legend()
# plt.show()

########################################################
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib as mpl
#
# # 每个数据集在不同长度范围内的样本数量
# # dataset1_counts = [20, 35, 45, 30, 10, 5, 2]  # 示例数据，可以替换为实际数据
# # dataset2_counts = [15, 25, 40, 20, 8, 4, 1]  # 示例数据，可以替换为实际数据
# # dataset3_counts = [10, 20, 30, 25, 12, 6, 3]  # 示例数据，可以替换为实际数据
# tnews = [11293, 38574, 3491, 1, 1, 0, 0]
# cnews = [2, 161, 662, 3124, 4851, 9835, 31365]
# csldcp = [1, 2, 64, 430, 1030, 466, 43]
# # 计算每个数据集的总样本数
# dataset1_total = sum(tnews)
# dataset2_total = sum(cnews)
# dataset3_total = sum(csldcp)
#
# # 计算每个数据集在范围内的样本数量占总样本数的百分比
# dataset1_percentages = [count / dataset1_total * 100 for count in tnews]
# dataset2_percentages = [count / dataset2_total * 100 for count in cnews]
# dataset3_percentages = [count / dataset3_total * 100 for count in csldcp]
#
# # 定义长度范围和范围标签
# bins = ['0-16', '16-32', '32-64', '64-128', '128-256', '256-512', '512-']
#
# # 绘制直方图
# fig, ax = plt.subplots(dpi=300)
# x = np.arange(len(bins))
# width = 0.3
#
# rects1 = ax.bar(x - width, dataset1_percentages, width, label='Tnews')
# rects2 = ax.bar(x, dataset2_percentages, width, label='THUCNews')
# rects3 = ax.bar(x + width, dataset3_percentages, width, label='CSLDCP')
#
# # 设置坐标轴和标题
# ax.set_xlabel('长度范围')
# ax.set_ylabel('样本数量百分比')
# ax.set_title('不同数据集样本长度分布直方图')
# ax.set_xticks(x)
# ax.set_xticklabels(bins)
# ax.legend()
#
# # 添加数据标签
# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         if height > 1:
#             ax.annotate('{:.1f}%'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom',fontsize=8)
#
# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
#
# # # 正确显示中文和负号
# # plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
# # mpl.rcParams['font.family'] = 'sans-serif'
# # mpl.rcParams['font.sans-serif'] = ['NSimSun']
# plt.rcParams["axes.unicode_minus"] = False
# # 显示图形
# plt.show()