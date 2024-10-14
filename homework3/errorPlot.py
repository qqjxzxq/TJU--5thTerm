import matplotlib.pyplot as plt
import numpy as np

# 支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 数据
orders = [1, 2, 3, 4]
divisions = [2, 4, 8]
errors = np.array([
    [45.9391504806, 16.2939150346, 0.598382161426, 0.335934968383],
    [7.83728249188, 0.4562569593188, 0.290639062674, 0.283680315687],
    [0.691849651851, 0.2576461305532, 0.2337412440088, 0.2022092773411]
])

# 绘制图表
fig, ax = plt.subplots()
for i, division in enumerate(divisions):
    ax.plot(orders, errors[i], marker='o', label=f'{division} 等分')

# 设置标题和标签
ax.set_title('不同等分区间和多项式阶次的拟合误差')
ax.set_xlabel('多项式阶次')
ax.set_ylabel('误差')
ax.legend(title='等分区间数')

# 显示网格
ax.grid(True)

# 展示图表
plt.show()
