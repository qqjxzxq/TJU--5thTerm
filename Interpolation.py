import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sin, cos, lambdify

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

class Interpolation:
    def __init__(self, start, end, num_points, func_params, exp_points):
        self.start = start
        self.end = end
        self.num_points = num_points
        self.func_params = func_params
        self.exp_points = exp_points
        self.t = symbols('t')  # 定义符号变量
        self.x = self.chebyshev_nodes()
        self.y = self.compute_function_values()
    
    #使用chebyshev采样n+1个点    
    def chebyshev_nodes(self):
        nodes = []
        for i in range(self.num_points):
            x_i = 0.5 * (self.start + self.end) + 0.5 * (self.end - self.start) * np.cos((2 * i + 1) / (2 * self.num_points) * np.pi)
            nodes.append(x_i)
        return np.array(nodes)

    def compute_function_values(self):
        a, b, c, d = self.func_params
        func = lambdify(self.t, int(a) * sin(int(b) * self.t) + int(c) * cos(int(d) * self.t))
        return func(self.x)

    def display_points(self):
        points = list(zip(self.x, self.y))
        # for point in points:
        #     print(f"点: x = {point[0]}, y = {point[1]}")
            
    
    #Vandermonde
#def Vandermonde()


#Lagrange



#Newton


#差分 Newton


#分段线性插值



#分段三次Hermite插值


############# 程序入口 ############

a, b = input("输入插值区间：").split()
c, d, e, f = input("输入函数参数：").split()
n = int(input("输入采样个数 n ："))
m = int(input("输入实验点个数 m ："))

# 构造插值对象
interp = Interpolation(start=int(a), end=int(b), num_points=n + 1, func_params=(c, d, e, f), exp_points= m)

# # 显示生成的点
# interp.display_points()




