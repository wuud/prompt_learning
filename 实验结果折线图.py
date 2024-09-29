import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt


labels = [1, 4, 8, 16]
x= [1,2,3,4]

FT1 = [54.6, 81.7, 88, 92]
FT1_e = [13.1, 6.4, 4.8, 1.1]
PT1 = [63, 74.7, 80.3, 82.2]
PT1_e = [2.8, 3.0, 2.3, 1.7]
AUTO1 = [42.9, 74.9, 81.3, 88.6]
AUTO1_e = [4.6, 2.9, 4.1, 1.4]
SOFT1 = [53.8, 81.1, 92, 94.1]
SOFT1_e = [3.3, 6.5, 2.0, 0.1]
Our1 = [79.6, 86.9, 90.3, 94.2]
Our1_e = [1.5, 4.8, 4.4, 1.2]

FT2 = [25.6, 41.7, 46.9, 50.3]
FT2_e = [2.4, 4.8, 3.9, 1.4]
PT2 = [40.2, 49.9, 51.3, 54.3]
PT2_e = [1.7, 1.2, 1.9, 1.1]
AUTO2 = [22.8, 33.8, 48.1, 51.1]
AUTO2_e = [2.7, 1.7, 3.2, 2.2]
SOFT2 = [21.0, 36.7, 45.3, 51.8]
SOFT2_e = [3.5, 6.5, 3.6, 0.2]
Our2 = [55.4, 55.4, 56.6, 57.6]
Our2_e = [1.0, 0.5, 0.5, 0.6]

FT3 = [26.1, 42.8, 48.5, 52.8]
FT3_e = [2.2, 0.6, 1.8, 0.9]
PT3 = [38.2, 44.3, 46.8, 49.4]
PT3_e = [0.3, 1.0, 0.6, 0.8]
AUTO3 = [16.2, 37.1, 43.8, 49.3]
AUTO3_e = [3.7, 3.2, 2.7, 0.9]
SOFT3 = [17.3, 40.9, 52.0, 52.9]
SOFT3_e = [2.9, 2.1, 0.5, 0.8]
Our3 = [48.4, 51.2, 53.3, 56]
Our3_e = [0.4, 0.7, 0.6, 0.3]

# 创建画布和子图
plt.xticks(x, labels)
# fig, axs = plt.subplots(2, 2)
fig = plt.figure(figsize=(10, 12))  # 设置整个图的大小
gs = gridspec.GridSpec(2, 15, height_ratios=[1,1])  # 设置网格布局和每行的高度比例

# 绘制第一个小图
ax1 = fig.add_subplot(gs[0, :7])
ax1.plot(x, FT2, label='FT', marker='o')
ax1.fill_between(x, np.subtract(FT2, FT2_e), np.add(FT2, FT2_e), alpha=0.3)
ax1.plot(x, PT2, label='PL', marker='s')
ax1.fill_between(x, np.subtract(PT2, PT2_e), np.add(PT2, PT2_e), alpha=0.3)

ax1.plot(x, SOFT2, label='SOFT', marker='^')
ax1.fill_between(x, np.subtract(SOFT2, SOFT2_e), np.add(SOFT2, SOFT2_e), alpha=0.3)

ax1.plot(x, Our2, label='Our', marker='h')
ax1.fill_between(x, np.subtract(Our2, Our2_e), np.add(Our2, Our2_e), alpha=0.3)

ax1.plot(x, AUTO2, label='AUTO', marker='d')
ax1.fill_between(x, np.subtract(AUTO2, AUTO2_e), np.add(AUTO2, AUTO2_e), alpha=0.3)

ax1.set_title('Tnews')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_xlabel('Shot')
ax1.set_ylabel('Accuracy')
# ax1.set_yticks([20, 25, 30, 35, 40, 45, 50, 55])  # 设置纵坐标刻度位置
# ax1.set_yticklabels(['20', '25', '30', '35', '40', '45', '50', '55'])  # 设置纵坐标刻度标签
ax1.legend()

# 绘制第二个小图
ax2 = fig.add_subplot(gs[0, 8:])

ax2.plot(x, FT1, label='FT', marker='o')
ax2.fill_between(x, np.subtract(FT1, FT1_e), np.add(FT1, FT1_e), alpha=0.3)

ax2.plot(x, PT1, label='PL', marker='s')
ax2.fill_between(x, np.subtract(PT1, PT1_e), np.add(PT1, PT1_e), alpha=0.3)

ax2.plot(x, SOFT1, label='SOFT', marker='^')
ax2.fill_between(x, np.subtract(SOFT1, SOFT1_e), np.add(SOFT1, SOFT1_e), alpha=0.3)

ax2.plot(x, Our1, label='Our', marker='h')
ax2.fill_between(x, np.subtract(Our1, Our1_e), np.add(Our1, Our1_e), alpha=0.3)

ax2.plot(x, AUTO1, label='AUTO', marker='d')
ax2.fill_between(x, np.subtract(AUTO1, AUTO1_e), np.add(AUTO1, AUTO1_e), alpha=0.3)

ax2.set_title('THUCNews')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_xlabel('Shot')
ax2.set_ylabel('Accuracy')
ax2.legend()

# 绘制第三个小图
ax3 = fig.add_subplot(gs[1, 4:11])
ax3.plot(x, FT3, label='FT', marker='o')
ax3.fill_between(x, np.subtract(FT3, FT3_e), np.add(FT3, FT3_e), alpha=0.3)

ax3.plot(x, PT3, label='PL', marker='s')
ax3.fill_between(x, np.subtract(PT3, PT3_e), np.add(PT3, PT3_e), alpha=0.3)

ax3.plot(x, SOFT3, label='SOFT', marker='^')
ax3.fill_between(x, np.subtract(SOFT3, SOFT3_e), np.add(SOFT3, SOFT3_e), alpha=0.3)

ax3.plot(x, Our3, label='Our', marker='h')
ax3.fill_between(x, np.subtract(Our3, Our3_e), np.add(Our3, Our3_e), alpha=0.3)

ax3.plot(x, AUTO3, label='AUTO', marker='d')
ax3.fill_between(x, np.subtract(AUTO3, AUTO3_e), np.add(AUTO3, AUTO3_e), alpha=0.3)

ax3.set_title('CSLDCP')
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.set_xlabel('Shot')
ax3.set_ylabel('Accuracy')
ax3.legend()

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
fig.suptitle("不同方法在三个数据集上的表现", fontsize='18')

dpi = 300
fig.set_dpi(dpi)

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()