# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import random
from sympy import symbols, lambdify, sin, cos

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

class CurveFitting:
    def __init__(self, start, end, num_points, func_params, exp_points, order, noise_level, max_iters=1000, sigma=0.5):
        self.start = start
        self.end = end
        self.num_points = num_points
        self.func_params = func_params
        self.exp_points = exp_points
        self.order = order  
        self.noise_level = noise_level
        self.max_iters = max_iters  
        self.sigma = sigma  

        # 初始化：生成实验点，拟合并可视化
        self.random_m_exp()
        self.ransac()  
        self.calculate_errors()  # 计算误差
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

    def ransac(self):
        best_coefficients = None
        max_inliers = 0

        for _ in range(self.max_iters):
            sample_indices = random.sample(range(self.exp_points), self.order + 1)
            x_sample = [self.m_node_x[i] for i in sample_indices]
            y_sample = [self.m_node_y_noisy[i] for i in sample_indices]

            coefficients = np.polyfit(x_sample, y_sample, self.order)

            inliers = 0
            for i in range(self.exp_points):
                y_estimated = np.polyval(coefficients, self.m_node_x[i])
                if abs(y_estimated - self.m_node_y_noisy[i]) < self.sigma:
                    inliers += 1

            if inliers > max_inliers:
                max_inliers = inliers
                best_coefficients = coefficients

        self.coefficients = best_coefficients
        print("拟合多项式的系数：", self.coefficients)

        t = symbols('t')
        self.fit_func = lambdify(t, sum(c * (t ** i) for i, c in enumerate(reversed(self.coefficients))))

    def calculate_errors(self):
        y_fit_noisy = [self.fit_func(x) for x in self.m_node_x]
        errors = [(y_noisy - y_fit_val) ** 2 for y_noisy, y_fit_val in zip(self.m_node_y_noisy, y_fit_noisy)]
        
        self.sse_noisy = sum(errors)
        self.mse_noisy = self.sse_noisy / len(self.m_node_x)

        print(f"[RANSAC] 均方误差 (MSE)：{self.mse_noisy}")
        print(f"[RANSAC] 平方误差和 (SSE)：{self.sse_noisy}")

    def plot(self):
        x_vals = np.linspace(self.start, self.end, 100)
        y_vals = [self.func(x) for x in x_vals]
        y_fit_vals = [self.fit_func(x) for x in x_vals]

        plt.plot(x_vals, y_vals, 'r-', label='原始函数')
        plt.plot(x_vals, y_fit_vals, 'g--', label='RANSAC拟合')
        plt.scatter(self.m_node_x, self.m_node_y_noisy, color='c', label='实验点 (有扰动)', marker='x')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('RANSAC 最小二乘法拟合')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        start, end = int(sys.argv[1]), int(sys.argv[2])
        a, b, c, d = map(int, sys.argv[3:7])
        num_points = int(sys.argv[7])
        exp_points = int(sys.argv[8])
        order = int(sys.argv[9])
        noise_level = float(sys.argv[10])
    else:
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
