import os
import numpy as np
import matplotlib.pyplot as plt
from CurveFitting2 import CurveFitting  # 假设主程序保存为 CurveFitting.py

# 创建保存图片的目录
save_dir = "/Users/tanqianjian/Desktop/数值计算/homework3/leastsq_picture2"
os.makedirs(save_dir, exist_ok=True)

# 批量测试的参数配置
intervals = [(-1, 1)]  # 插值区间
function_params = [(1, 2, 3, 4)]  # 不同的函数参数
num_samples = [20]  # 采样点数 n
num_experiments = [100]  # 实验点数 m

# 开始批量测试
for interval in intervals:
    for params in function_params:
        for n in num_samples:
            for m in num_experiments:
                # 打印测试信息
                print(f"正在测试区间 {interval}，函数参数 {params}，采样点数 {n}，实验点数 {m}...")

                # 创建新图表
                plt.figure()

                # 执行曲线拟合
                curve_fitting = CurveFitting(
                    start=interval[0], end=interval[1],
                    num_points=n + 1, func_params=params,
                    exp_points=m
                )

                # 构造文件名并保存图片
                filename = f"fit_{interval}_{params}_n{n}_m{m}.png".replace(' ', '')
                filepath = os.path.join(save_dir, filename)

                plt.savefig(filepath)  # 保存图片，高分辨率和紧凑布局
                plt.close()  # 关闭当前图表

print("所有批量测试完成！") 
