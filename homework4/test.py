import os
from NumericalIntegration import NumericalIntegration  # 导入类

# 保存结果的文件路径
output_file = "/Users/tanqianjian/Desktop/数值计算/homework4/integration_results.txt"

# 定义批量测试参数 (a, b, h, precision)
test_cases = [
    (1, 20, 1, 1e-3),
    (1, 20, 1, 1e-4),
    (1, 20, 1, 1e-5),
    (1, 20, 1, 1e-6),
]

def batch_test():
    with open(output_file, 'w') as f:
        f.write(f"{'Test Case':<10} {'区间 (a,b)':<15} {'步长 h':<10} {'复合梯形法积分值':<30} {'划分次数':<10} {'Romberg法积分值':<30} {'划分次数':<10}\n")
        f.write("-" * 150 + "\n")

        # 遍历每个测试用例，执行积分计算
        for idx, (a, b, h, precision) in enumerate(test_cases):
            print(f"Running Test Case {idx + 1}...")
            integrator = NumericalIntegration(a=a, b=b, h=h, precision=precision)

            # 复合梯形法
            trapezoidal_result = integrator.trapezoidal_rule(int((b - a) / h))
            trapezoidal_n = int((b - a) / h)

            # 龙贝格法
            romberg_result, romberg_n, romberg_h = integrator.romberg()

            # 记录结果
            f.write(f"{idx + 1:<10} {a:.2f}, {b:.2f}      {h:<10} {trapezoidal_result:<30} {trapezoidal_n:<10} {romberg_result:<30} {romberg_n:<10}\n")

if __name__ == "__main__":
    batch_test()
