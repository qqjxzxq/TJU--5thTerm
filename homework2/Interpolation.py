import numpy as np
import matplotlib.pyplot as plt
import math
from sympy import symbols, sin, cos, lambdify
import random

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

class Interpolation:
    def __init__(self, start, end, num_points, func_params, exp_points):
        self.start = start
        self.end = end
        self.num_points = num_points
        self.func_params = func_params
        self.points = None
        self.exp_points = exp_points
        self.pi = np.pi

        self.t = symbols('t')  # 定义符号变量
        self.x, self.y = self.UniformSampling() # 1 使用均匀采样
        #self.x, self.y = self.chebyshev_sampling() # 2 使用切比雪夫采样
        self.Vandermonde = self.VandermondeError() # 计算范德蒙德多项式插值及其误差
        
    def func(self, x):
        a, b, c, d = self.func_params 
        fx = int(a) * np.sin(int(b) * x) + int(c) * np.cos(int(d) * x) 
        return fx
        
    #使用均匀采样n+1个点
    def UniformSampling(self):
        
        self.x = np.linspace(self.start, self.end, self.num_points)
        self.y = self.func(self.x)
        
        self.points = list(zip(self.x, self.y))
        for point in self.points:
            print(f"点: x = {point[0]}, y = {point[1]}")
        
        return self.x, self.y
    
    
    #使用chebyshev采样n+1个点    
    def chebyshev_sampling(self):
        nodes = []
        for i in range(self.num_points):
            x_i = 0.5 * (self.start + self.end) + 0.5 * (self.end - self.start) * np.cos((2 * i + 1) / (2 * self.num_points) * np.pi)
            nodes.append(x_i)
    
        self.x = np.array(nodes)
        self.y = func(self.x)
    
        self.points = list(zip(self.x, self.y))
        for point in self.points:
            print(f"点: x = {point[0]}, y = {point[1]}")
            
        return self.x, self.y

        
#Vandermonde
    def Vandermonde(self, x):
    # 初始化 Vandermonde 矩阵
        X = np.zeros((self.num_points, self.num_points))
        for i in range(self.num_points):
            for j in range(self.num_points):
                X[i][j] = np.power(self.x[i], j)

    # 构造系数矩阵 Y
        Y = np.array(self.y).reshape(-1, 1)  # 转换为列向量

    # 解线性方程组
        A = np.linalg.solve(X, Y)

    # 计算多项式值 P
        P = 0
        
        for i in range(self.num_points):
            P += A[i][0] * np.power(x, i)
            print("插值结果：x = ", self.x[i] , ", y = ", P)
            
        return P
    

    def VandermondeError(self):
        #随机选取m个点
        N = range(self.start, self.end)
        node = random.sample(N, self.exp_points)
        Rn = []
        for i in range(self.exp_points):  
            error = self.func(node[i]) - self.Vandermonde(node[i])
            Rn.append(error)
            print("x = " , node[i] , "误差R = " , Rn[i])
        
        mean = np.mean(Rn)
        print("范德蒙德多项式插值平均误差：", mean)
            
    


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






