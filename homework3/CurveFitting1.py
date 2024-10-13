# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import random
from sympy import symbols, lambdify, sin, cos

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
class CurveFitting:
    def __init__(self, start, end, num_points, func_params, exp_points, order, noise_level):
        self.start = start
        self.end = end
        self.num_points = num_points
        self.func_params = func_params
        self.exp_points = exp_points
        self.order = order  # 多项式阶次
        self.noise_level = noise_level

        # 初始化：生成实验点，拟合并可视化
        self.random_m_exp()
        self.fit_func, self.coefficients = self.leastsq()
        print("拟合多项式的系数：", self.coefficients)

        self.calculate_errors()
        self.calculate_noisy_errors()  # 计算带噪声的误差
        self.plot()

    def func(self, x):
        a, b, c, d = self.func_params
        return int(a) * np.sin(int(b) * x) + int(c) * np.cos(int(d) * x)

    def random_m_exp(self):
        N = np.linspace(self.start, self.end, max(self.exp_points * 10, self.exp_points))
        self.m_node_x = sorted(random.sample(list(N), self.exp_points))
        self.m_node_y = [self.func(x) for x in self.m_node_x]

        # 添加噪声
        noise = np.random.normal(0, self.noise_level, self.exp_points)
        self.m_node_y_noisy = [y + n for y, n in zip(self.m_node_y, noise)]

    def leastsq(self):
        matA = np.zeros((self.order + 1, self.order + 1))
        matB = np.zeros(self.order + 1)

        for i in range(self.order + 1):
            for j in range(self.order + 1):
                matA[i, j] = sum(x ** (i + j) for x in self.m_node_x)
            matB[i] = sum(y * (x ** i) for x, y in zip(self.m_node_x, self.m_node_y))

        coefficients = np.linalg.solve(matA, matB)
        t = symbols('t')
        fit_func = sum(coeff * (t ** i) for i, coeff in enumerate(coefficients))
        return lambdify(t, fit_func), coefficients

    def calculate_errors(self):
        y_fit = [self.fit_func(x) for x in self.m_node_x]
        errors = [(y - y_fit_val) ** 2 for y, y_fit_val in zip(self.m_node_y, y_fit)]
        self.sse = sum(errors)
        self.mse = self.sse / len(self.m_node_x)
        print(f"[无扰动] 均方误差 (MSE)：{self.mse}")
        print(f"[无扰动] 平方误差和 (SSE)：{self.sse}")

    def calculate_noisy_errors(self):
        y_fit_noisy = [self.fit_func(x) for x in self.m_node_x]  # 拟合噪声后的数据
        errors_noisy = [(y_noisy - y_fit_val) ** 2 for y_noisy, y_fit_val in zip(self.m_node_y_noisy, y_fit_noisy)]
        self.sse_noisy = sum(errors_noisy)
        self.mse_noisy = self.sse_noisy / len(self.m_node_x)
        print(f"[有扰动] 均方误差 (MSE)：{self.mse_noisy}")
        print(f"[有扰动] 平方误差和 (SSE)：{self.sse_noisy}")

    def plot(self):
        x_vals = np.linspace(self.start, self.end, 100)
        y_vals = [self.func(x) for x in x_vals]
        y_fit_vals = [self.fit_func(x) for x in x_vals]
        y_fit_noisy_vals = [self.fit_func(x) for x in x_vals]  # 使用相同的拟合函数

        plt.plot(x_vals, y_vals, 'r-', label='原始函数')
        plt.plot(x_vals, y_fit_vals, 'g--', label='无扰动拟合')
        plt.plot(x_vals, y_fit_noisy_vals, 'b-.', label='有扰动拟合')  # 使用相同的拟合函数
        plt.scatter(self.m_node_x, self.m_node_y, color='m', label='实验点 (无扰动)')
        plt.scatter(self.m_node_x, self.m_node_y_noisy, color='c', label='实验点 (有扰动)', marker='x')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('最小二乘法多项式拟合')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    import sys

    # 检查是否有命令行参数传入，用于批量测试
    if len(sys.argv) > 1:
        start, end = int(sys.argv[1]), int(sys.argv[2])
        a, b, c, d = map(int, sys.argv[3:7])
        num_points = int(sys.argv[7])
        exp_points = int(sys.argv[8])
        order = int(sys.argv[9])
        noise_level = float(sys.argv[10])
    else:
        # 命令行输入
        start, end = map(int, input("输入插值区间 (如 -1 1)：").split())
        a, b, c, d = map(int, input("输入函数参数 (如 1 2 3 4)：").split())
        num_points = int(input("输入采样个数 n：")) + 1
        exp_points = int(input("输入实验点个数 m："))
        order = int(input("输入多项式阶次："))
        noise_level = float(input("输入扰动幅度 (如 0.1)："))

    curve_fitting = CurveFitting(
        start=start, end=end, num_points=num_points,
        func_params=(a, b, c, d), exp_points=exp_points, 
        order=order, noise_level=noise_level
    )
