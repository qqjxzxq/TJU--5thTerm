#应用到numpy和pandas模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_excel(r'D:\桌面\研一文件\课程学习\数值计算\Python\线性拟合.xlsx')
data.head()
x_list = data.x.to_list()
y_list = data.y.to_list()

# x_array,y_array是我们要拟合的数据
x_array = np.array(x_list)
y_array = np.array(y_list)
# 方程个数
m = len(x_array)
# 计算过程
sum_x = np.sum(x_array)
sum_y = np.sum(y_array)
sum_xy = np.sum(x_array * y_array)
sum_xx = np.sum(x_array ** 2)
a = (sum_y * sum_xx - sum_x * sum_xy) / (m * sum_xx - (sum_x) ** 2)
b = (m * sum_xy - sum_x * sum_y) / (m * sum_xx - (sum_x) ** 2)
print("p = {:.4f} + {:.4f}x".format(a,b))
X_n = np.linspace(1, 6, 10)
Y_n = a + b * X_n
l1, = plt.plot(X_n, Y_n)
plt.scatter(x_list, y_list, color='blue', marker='.')
plt.legend(handles=[l1], labels=['g(x)'], loc='best')
plt.show()