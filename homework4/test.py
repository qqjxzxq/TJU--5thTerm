# coding=utf-8
import os
from CurveFitting1 import CurveFitting  # 导入类
import matplotlib.pyplot as plt
import numpy as np

# 保存图片路径
save_dir = "/Users/tanqianjian/Desktop/数值计算/homework3/leastsq_picture1"
os.makedirs(save_dir, exist_ok=True)  # 自动创建目录

def batch_test():
    # 定义批量测试参数 (start, end, func_params, num_points, exp_points, order, noise_level)
    test_cases = [
        (1, 10, (1, 1, 1, 1), 10, 50, 1, 0.1),
        (1, 10, (1, 1, 1, 1), 10, 50, 2, 0.1),
        (1, 10, (1, 1, 1, 1), 10, 50, 3, 0.1),
        (1, 10, (1, 1, 1, 1), 10, 50, 4, 0.1),
        (-1, 1, (1, 2, 3, 4), 20, 100, 1, 0.1),
        (-1, 1, (1, 2, 3, 4), 20, 100, 2, 0.1),
        (-1, 1, (1, 2, 3, 4), 20, 100, 3, 0.1),
        (-1, 1, (1, 2, 3, 4), 20, 100, 4, 0.1),
    ]

    print(f"{'Test Case':<10} {'MSE (无扰动)':<15} {'SSE (无扰动)':<15} {'MSE (有扰动)':<15} {'SSE (有扰动)':<15}")
    print("-" * 75)

    # 遍历每个测试用例，执行拟合并输出误差
    for idx, (start, end, func_params, n, m, order, noise_level) in enumerate(test_cases):
        print(f"Running Test Case {idx + 1}...")
        
        # 初始化 CurveFitting 类
        curve_fitting = CurveFitting(
            start=start, end=end, num_points=n + 1,
            func_params=func_params, exp_points=m, order=order, noise_level=noise_level
        )

        # 获取数据点用于绘图
        x_vals = np.linspace(start, end, 100)  # 在区间内生成100个x值
        y_vals = [curve_fitting.func(x) for x in x_vals]  # 原始函数的y值
        y_fit_vals = [curve_fitting.fit_func(x) for x in x_vals]  # 拟合函数的y值
        y_fit_vals_noisy = y_fit_vals  # 有扰动的拟合值，直接使用无扰动拟合函数

        # 绘制并保存图像
        plt.plot(x_vals, y_vals, 'r-', label='原始函数')
        plt.plot(x_vals, y_fit_vals, 'g--', label='无扰动拟合')
        plt.plot(x_vals, y_fit_vals_noisy, 'b-.', label='有扰动拟合')  # 使用相同的拟合函数
        plt.scatter(curve_fitting.m_node_x, curve_fitting.m_node_y, color='m', label='实验点 (无扰动)')
        plt.scatter(curve_fitting.m_node_x, curve_fitting.m_node_y_noisy, color='c', label='实验点 (有扰动)', marker='x')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'最小二乘法拟合 - Test Case {idx + 1}')
        plt.grid(True)

        # 保存图像到指定目录
        save_path = os.path.join(save_dir, f"test_case_{idx + 1}.png")
        plt.savefig(save_path)
        plt.close()  # 关闭当前绘图，避免影响后续图像

        # 输出误差信息
        print(f"{idx + 1:<10} {curve_fitting.mse:<15.5f} {curve_fitting.sse:<15.5f} {curve_fitting.mse_noisy:<15.5f} {curve_fitting.sse_noisy:<15.5f}")
        print()  # 换行以分隔测试用例

if __name__ == "__main__":
    batch_test()
