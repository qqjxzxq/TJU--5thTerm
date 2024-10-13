# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import random
from sympy import symbols, lambdify, sin, cos

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

class CurveFitting:
    def __init__(self, start, end, num_points, func_params, exp_points):
        self.start = start
        self.end = end
        self.num_points = num_points
        self.func_params = func_params
        self.exp_points = exp_points
        self.pi = np.pi

        # 初始化：生成实验点并拟合多次曲线
        self.random_m_exp()  # 生成实验点
        self.subdivision_and_fitting()  # 进行多次拟合

    # 目标函数
    def func(self, x):
        a, b, c, d = self.func_params
        fx = int(a) * np.sin(int(b) * x) + int(c) * np.cos(int(d) * x)
        return fx

    # 随机生成 m 个实验点（并添加 Y 值扰动）
    def random_m_exp(self):
        N = np.linspace(self.start, self.end, max(self.exp_points * 10, self.exp_points))
        self.m_node_x = sorted(random.sample(list(N), self.exp_points))
        self.m_node_y = [self.func(x) for x in self.m_node_x]

        # 添加较小的随机扰动
        self.m_node_y_perturbed = [y + random.uniform(-0.1, 0.1) for y in self.m_node_y]

    # 最小二乘法拟合（不调用库）
    def leastsq(self, x_vals, y_vals, order):
        matA = np.zeros((order + 1, order + 1))
        matB = np.zeros(order + 1)

        # 构造矩阵 A 和向量 B
        for i in range(order + 1):
            for j in range(order + 1):
                matA[i, j] = sum(x ** (i + j) for x in x_vals)
            matB[i] = sum(y * (x ** i) for x, y in zip(x_vals, y_vals))

        # 解线性方程组 A * coeffs = B，得到多项式系数
        coefficients = np.linalg.solve(matA, matB)

        # 生成拟合函数
        t = symbols('t')
        fit_func = sum(coeff * (t ** i) for i, coeff in enumerate(coefficients))
        return lambdify(t, fit_func), coefficients

    # 对区间进行 2、4、8 等分并拟合不同阶次的多项式
    def subdivision_and_fitting(self):
        subdivisions = [2, 4, 8]  # 2、4、8 等分
        max_order = 4  # 多项式的最高阶次

        for div in subdivisions:
            sub_intervals = np.linspace(self.start, self.end, div + 1)
            print(f"\n=== {div} 等分区间的拟合结果 ===")

            for order in range(1, max_order + 1):
                print(f"\n使用 {order} 次多项式拟合每个子区间：")
                total_error = 0  # 用于累计每个区间的误差

                for i in range(div):
                    # 获取当前子区间的实验点
                    x_sub = [x for x in self.m_node_x if sub_intervals[i] <= x < sub_intervals[i + 1]]
                    y_sub = [y for x, y in zip(self.m_node_x, self.m_node_y) if sub_intervals[i] <= x < sub_intervals[i + 1]]
                    y_sub_perturbed = [y for x, y in zip(self.m_node_x, self.m_node_y_perturbed) if sub_intervals[i] <= x < sub_intervals[i + 1]]

                    if len(x_sub) < order + 1:
                        print(f"  - 子区间 [{sub_intervals[i]}, {sub_intervals[i + 1]}] 的实验点不足，跳过")
                        continue

                    # 执行拟合（扰动前和扰动后）
                    fit_func, _ = self.leastsq(x_sub, y_sub, order)
                    fit_func_perturbed, _ = self.leastsq(x_sub, y_sub_perturbed, order)

                    # 计算误差
                    error = sum((fit_func(x) - y) ** 2 for x, y in zip(x_sub, y_sub))
                    total_error += error

                    # 可视化子区间拟合
                    self.plot(sub_intervals[i], sub_intervals[i + 1], fit_func, fit_func_perturbed, x_sub, y_sub)

                print(f"  - {order} 次多项式的总误差：{total_error}")

    # 可视化拟合结果（添加原始目标函数曲线）
    def plot(self, start, end, fit_func, fit_func_perturbed, x_sub, y_sub):
        x_vals = np.linspace(start, end, 100)
        y_true = [self.func(x) for x in x_vals]  # 原始目标函数
        y_fit = [fit_func(x) for x in x_vals]
        y_fit_perturbed = [fit_func_perturbed(x) for x in x_vals]

        plt.plot(x_vals, y_true, 'r-', label='原始目标函数')  # 绘制原始函数
        plt.plot(x_vals, y_fit, 'g--', label='拟合曲线（无扰动）')
        plt.plot(x_vals, y_fit_perturbed, 'b-.', label='拟合曲线（有扰动）')
        plt.scatter(x_sub, y_sub, color='m', label='实验点')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'区间 [{start}, {end}] 的多项式拟合')
        plt.grid(True)
        plt.show()


# 程序入口
a, b = input("输入插值区间 (如 -1 1)：").split()
c, d, e, f = input("输入函数参数 (如 1 2 3 4)：").split()
n = int(input("输入采样个数 n："))
m = int(input("输入实验点个数 m："))

# 构造插值对象，并自动完成实验点生成、拟合和可视化
curve_fitting = CurveFitting(
    start=int(a), end=int(b), num_points=n + 1,
    func_params=(c, d, e, f), exp_points=m
)
