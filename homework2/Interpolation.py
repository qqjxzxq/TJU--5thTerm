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
        
        self.m_node_x = np.zeros(self.exp_points)
        self.m_Vandermonde_y = np.zeros(self.exp_points)
        self.m_lagrange_y = np.zeros(self.exp_points)
        self.m_Newton_y = np.zeros(self.exp_points)
        
        self.MEAN = np.zeros(5) #五种插值方法的平均误差

        self.x, self.y = self.UniformSampling() # 1 使用均匀采样
        #self.x, self.y = self.chebyshev_sampling() # 2 使用切比雪夫采样
        self.Random_m_exp = self.Random_m_exp() # 随机采样m个实验点
        self.VandermondeResult = self.VandermondeError() # 计算范德蒙德多项式插值及其误差
        self.lagrangeResult = self.lagrangeError() # 计算拉格朗日插值及其误差
        self.NewtonResult = self.NewtonError() # 计算牛顿插值及其误差 
        self.plot = self.plot_interpolation_vs_target() # 画图
        
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
            print(f"采样点: x = {point[0]}, 标准值：y = {point[1]}")
        
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
    
    
    #随机采样m个实验点
    def Random_m_exp(self):
        #随机选取m个点
        N = np.linspace(self.start, self.end, max(self.exp_points * 10, self.exp_points))
        self.m_node_x = random.sample(list(N), self.exp_points)
        self.m_node_x = sorted(self.m_node_x) 

        
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
        
        print("范德蒙德插值结果：x = ", x , ", y = ", P)
            
        return P
    

    def VandermondeError(self):
        
        Rn = []
        for i in range(self.exp_points): 
            self.m_Vandermonde_y[i] = self.Vandermonde(self.m_node_x[i])
            error = abs(self.func(self.m_node_x[i]) - self.m_Vandermonde_y[i])
            Rn.append(error)
            print("x = " , self.m_node_x[i] , "误差R = " , Rn[i])
        
        mean = np.mean(Rn)
        self.MEAN[0] = mean
        print("范德蒙德多项式插值平均误差：", mean)
            
    


#Lagrange
    def lagrange(self, x):
    
    # 初始化插值多项式
        ln = 0
    
    # 计算拉格朗日插值多项式
        for k in range(self.num_points):
        # 拉格朗日基函数
            fm = 1
            fz = 1
            for i in range(self.num_points):
                if i == k:
                    continue
                fm *= (self.x[k] - self.x[i])
                fz *= (x - self.x[i])  # 注意这里是用 value 代替 t 进行插值点计算
            base = fz / fm
        
        # 累加到多项式中
            ln += (self.y[k] * base)
        
        print("拉格朗日插值结果：x = ", x , ", y = ", ln)
    
        return ln


    def lagrangeError(self):
        
        Rn = []
        for i in range(self.exp_points): 
            self.m_lagrange_y[i] = self.lagrange(self.m_node_x[i])
            error = abs(self.func(self.m_node_x[i]) - self.m_lagrange_y[i])
            Rn.append(error)
            print("x = " , self.m_node_x[i] , "误差R = " , Rn[i])
        
        mean = np.mean(Rn)
        self.MEAN[1] = mean
        print("拉格朗日多项式插值平均误差：", mean)

#Newton


    def Newton(self, x):
    # 1. 计算均差表
        length = self.num_points - 1
        self.df = [[0] * length for _ in range(length)]
    
    # 第一次差分
        for j in range(length):
            self.df[0][j] = (self.y[j] - self.y[j + 1]) / (self.x[j] - self.x[j + 1])
    
    # 高阶差分
        for i in range(1, length):
            for j in range(i, length):
                self.df[i][j] = (self.df[i - 1][j] - self.df[i - 1][j - 1]) / (self.x[j + 1] - self.x[j - i])
    
    # 2. 计算牛顿插值多项式在指定点 value 的值
        f_value = self.y[0]  # 插值初始值为 y[0]
    
        for k in range(1, self.num_points):
            tem = self.df[k - 1][k - 1]  # 获取均差表中的差分值
            product = 1
            for j in range(k):
                product *= (x - self.x[j])  # 构造 (value - x_j) 的乘积项
            f_value += tem * product  # 累加插值结果
        
        print("牛顿插值结果：x = ", x , ", y = ", f_value) 
        return f_value
    
    
    def NewtonError(self):
        Rn = []
        for i in range(self.exp_points): 
            self.m_Newton_y[i] = self.Newton(self.m_node_x[i])
            error = abs(self.func(self.m_node_x[i]) - self.m_Newton_y[i])
            Rn.append(error)
            print("x = " , self.m_node_x[i] , "误差R = " , Rn[i])
        
        mean = np.mean(Rn)
        self.MEAN[2] = mean
        print("牛顿插值平均误差：", mean)
        



#分段线性插值



#分段三次Hermite插值




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
        
        for i in range(self.exp_points):
            exp_y_target[i] = self.func(self.m_node_x[i])  # 目标函数值
            exp_y_interp1[i] = self.m_Vandermonde_y[i]  
            exp_y_interp2[i] = self.m_lagrange_y[i]
            exp_y_interp3[i] = self.m_Newton_y[i]

    # 3. 绘制目标函数的精细图像
        plt.figure(figsize=(10, 6))
        plt.plot(m_100_x, m_100_y_target, label='目标函数 (精细)', color='blue')

    # 4. 绘制在 m 个实验点上的插值函数
        # plt.plot(self.m_node_x, exp_y_interp1, 'o-', label='范德蒙德插值函数', color='red', linestyle='--')
        # plt.plot(self.m_node_x, exp_y_interp2, 'o-', label='拉格朗日插值函数', color='green', linestyle='-.')
        plt.plot(self.m_node_x, exp_y_interp3, 'o-', label='牛顿插值函数', color='yellow', linestyle='--')

    # 6. 图例和标题
        plt.legend()
        # plt.title('范德蒙德多项式插值')
        # plt.title('拉格朗日多项式插值') 
        plt.title('牛顿多项式插值') 
        plt.xlabel('x')
        plt.ylabel('函数值')
        plt.grid()
        plt.show()
        
        
        

############# 程序入口 ############

a, b = input("输入插值区间：").split()
c, d, e, f = input("输入函数参数：").split()
n = int(input("输入采样个数 n ："))
m = int(input("输入实验点个数 m ："))

# 构造插值对象
interp = Interpolation(start=int(a), end=int(b), num_points=n + 1, func_params=(c, d, e, f), exp_points= m)





