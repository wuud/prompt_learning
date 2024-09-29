import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
labels = [1, 4, 8, 16]



e3 = [79.36666666666666, 82.96666666666665, 83.7, 84.53333333333333, ]
e5 = [79.06666666666666, 82.93333333333334, 83.96666666666667, 84.8, ]
e1 = [79.43333333333334, 81.23333333333333, 83.7, 85.03333333333335, ]
our = [79.36666666666666, 82.96666666666665, 83.96666666666667, 85.03333333333335]
# d4 = [81.5, 84.1, 86.3, 87.4]

fig = plt.figure(figsize=(10, 10), dpi=300)  # 设置整个图的大小
#
plt.xticks(x, labels)
plt.plot(x, e3, label='ϵ = 0.3', marker='o')
plt.plot(x, e5, label='ϵ = 0.5', marker='s')
plt.plot(x, e1, label='ϵ = 1', marker='^')
plt.plot(x, our, label='Our', marker='h')

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.title('不同扰动半径下模型的表现', fontsize='18')
plt.legend()
plt.xlabel('Shot')
plt.ylabel('Accuracy')

plt.show()
