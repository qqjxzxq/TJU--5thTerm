import numpy as np
import matplotlib.pyplot as plt
import math
from sympy import symbols, sin, cos, lambdify
import random

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

class Runge:
    def __init__(self, start  , end  , num_points , exp_points):
        self.start = start
        self.end = end
        self.num_points = num_points
        self.points = None
        self.exp_points = exp_points
        self.pi = np.pi
        
        self.m_node_x = np.zeros(self.exp_points)
        self.m_Vandermonde_y = np.zeros(self.exp_points)
        
        self.MEAN = np.zeros(5) #五种插值方法的平均误差

        self.x, self.y = self.UniformSampling() # 1 使用随机采样
        self.x, self.y = self.chebyshev_sampling() # 2 使用切比雪夫采样
        self.Random_m_exp = self.Random_m_exp() # 随机采样m个实验点
        self.VandermondeResult = self.VandermondeError() # 计算范德蒙德多项式插值及其误差
        self.lagrangeResult = self.lagrangeError() # 计算拉格朗日插值及其误差
        self.NewtonResult = self.NewtonError() # 计算牛顿插值及其误差 
        self.PiecewiselinearResult = self.PiecewiselinearError() # 计算分段线性插值及误差  
        self.HermiteResult = self.HermiteError() # 计算Hermite插值及其误差
        
        self.plot = self.plot_interpolation_vs_target() # 画图
        
    def func(self, x):
        a, b, c, d = self.func_params 
        fx = int(a) * np.sin(int(b) * x) + int(c) * np.cos(int(d) * x) 
        return fx
        
    
    #使用chebyshev采样n+1个点    
    def chebyshev_sampling(self):
        nodes = []
        for i in range(self.num_points):
            x_i = 0.5 * (self.start + self.end) + 0.5 * (self.end - self.start) * np.cos((2 * i + 1) / (2 * self.num_points) * np.pi)
            nodes.append(x_i)
    
        self.x = np.array(nodes)
        self.y = self.func(self.x)
    
        self.points = list(zip(self.x, self.y))
        for point in self.points:
            print(f"点: x = {point[0]}, y = {point[1]}")
            
        return self.x, self.y
    
    
    #随机采样m个实验点
    def Random_m_exp(self):
        #随机选取m个点
        N = np.linspace(self.start, self.end, max(self.exp_points * 10, self.exp_points))
        self.m_node_x = random.sample(list(N), self.exp_points)
        self.m_node_x = sorted(self.m_node_x) 

        

    def plot_interpolation_vs_target(self):
    # 1. 选择 100 * m 个点对目标函数进行精细绘图
        m_100_x = np.linspace(self.start, self.end, max(self.exp_points * 100, self.exp_points))

    # 计算在这些点上的目标函数值
        m_100_y_target = np.zeros(m_100_x.size)
        for i in range(m_100_x.size):
            m_100_y_target[i] = self.func(m_100_x[i])  # 目标函数值

    # 2. 使用已有的 m 个实验点 self.m_node_x
        exp_y_target = np.zeros(self.exp_points)
        exp_y_interp1 = np.zeros(self.exp_points)
        exp_y_interp2 = np.zeros(self.exp_points) 
        exp_y_interp3 = np.zeros(self.exp_points) 
        exp_y_interp4 = np.zeros(self.exp_points)  
        exp_y_interp5 = np.zeros(self.exp_points)  
        
        for i in range(self.exp_points):
            exp_y_target[i] = self.func(self.m_node_x[i])  # 目标函数值
            exp_y_interp1[i] = self.m_Vandermonde_y[i]  
            exp_y_interp2[i] = self.m_lagrange_y[i]
            exp_y_interp3[i] = self.m_Newton_y[i]
            exp_y_interp4[i] = self.m_Piecewiselinear_y[i] 
            exp_y_interp5[i] = self.m_Hermite_y[i] 

    # 3. 绘制目标函数的精细图像
        plt.figure(figsize=(10, 6))
        plt.plot(m_100_x, m_100_y_target, label='目标函数 (精细)', color='blue')

    # 4. 绘制在 m 个实验点上的插值函数
        plt.plot(self.m_node_x, exp_y_interp1, 'o-', label='范德蒙德插值函数', color='red', linestyle='--')
        plt.plot(self.m_node_x, exp_y_interp2, 'o-', label='拉格朗日插值函数', color='green', linestyle='-.')
        plt.plot(self.m_node_x, exp_y_interp3, 'o-', label='牛顿插值函数', color='yellow', linestyle='--')
        plt.plot(self.m_node_x, exp_y_interp4, 'o-', label='分段线性插值函数', color='black', linestyle='-.')
        plt.plot(self.m_node_x, exp_y_interp5, 'o-', label='分段三次Hermite插值函数', color='pink', linestyle=':')


    # 6. 图例和标题
        plt.legend()
        # plt.title('范德蒙德多项式插值')
        # plt.title('拉格朗日多项式插值') 
        # plt.title('牛顿多项式插值')
        # plt.title('分段线性多项式插值') 
        # plt.title('分段三次Hermite插值')
        plt.title('各种插值对比函数曲线') 
        
        plt.xlabel('x')
        plt.ylabel('函数值')
        plt.grid()
        plt.show()
        
        
        

############# 程序入口 ############
n = int(input("输入采样个数 n ："))
m = int(input("输入实验点个数 m ："))

# 构造插值对象

runge = Runge(start = int(-1), end = int(1),num_points = n + 1, exp_points = m)



