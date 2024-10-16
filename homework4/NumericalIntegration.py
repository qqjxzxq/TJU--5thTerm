import math
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
class NumericalIntegration:
    def __init__(self, a, b, h, precision):
        """
        初始化类。
        :param a: 积分区间的起点
        :param b: 积分区间的终点
        :param h: 初始步长
        :param precision: 要求的精度 E
        """
        if a <= 0:
            raise ValueError("区间起点 a 必须大于 0，因为 ln(x) 在 x=0 不定义。")
        self.a = a
        self.b = b
        self.h = h
        self.precision = precision

    def func(self, x):
        """被积函数 f(x) = x^(1/2) * ln(x)"""
        return np.sqrt(x) * np.log(x)  # 使用 numpy 的向量化运算

    def trapezoidal_rule(self, n):
        """复合梯形公式计算积分。"""
        h = (self.b - self.a) / n  # 步长
        result = 0.5 * (self.func(self.a) + self.func(self.b))  # 两端点的贡献

        # 对中间节点求和
        for i in range(1, n):
            result += self.func(self.a + i * h)

        return result * h  # 返回积分结果

    def romberg(self, max_iter=10):
        """
        使用 Romberg 算法计算积分，逐步增加区间划分次数，直到满足精度要求。
        :param max_iter: 最大迭代次数（防止无限循环）
        """
        R = [[0] * (max_iter + 1) for _ in range(max_iter + 1)]  # 存储结果
        R[0][0] = self.trapezoidal_rule(1)  # 初始值

        for k in range(1, max_iter + 1):
            R[k][0] = self.trapezoidal_rule(2**k)  # 2^k 等分后的梯形积分
            for j in range(1, k + 1):
                R[k][j] = (4**j * R[k][j-1] - R[k-1][j-1]) / (4**j - 1)  # Romberg 递推公式

            # 检查是否达到精度要求
            if abs(R[k][k] - R[k-1][k-1]) < self.precision:
                return R[k][k], 2**k, (self.b - self.a) / 2**k  # 返回积分值、划分次数、步长

        raise ValueError("未能在最大迭代次数内达到所需精度")

    def compare_methods(self):
        """
        比较复合梯形公式和 Romberg 算法的积分结果、划分次数及步长大小。
        """
        # 使用给定步长 h 计算复合梯形公式
        n = int((self.b - self.a) / self.h)  # 根据步长 h 计算划分次数
        trapezoidal_result = self.trapezoidal_rule(n)

        # 使用 Romberg 算法计算
        romberg_result, romberg_n, romberg_h = self.romberg()

        # 输出结果对比
        print(f"复合梯形公式：积分值={trapezoidal_result}, 划分次数={n}, 步长={self.h}")
        print(f"Romberg算法：积分值={romberg_result}, 划分次数={romberg_n}, 步长={romberg_h}")

    def plot_function(self):
        """绘制被积函数和积分区域。"""
        x = np.linspace(self.a, self.b, 100)
        y = self.func(x)

        plt.plot(x, y, label='f(x) = x^(1/2) * ln(x)', color='blue')
        plt.fill_between(x, y, where=[(xi >= self.a and xi <= self.b) for xi in x], 
                        color='lightgreen', alpha=0.5, label='积分区域')

        plt.title('被积函数及积分区域')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
        plt.legend()
        plt.grid()
        plt.show()

def interactive_input():
    """命令行交互式输入参数。"""
    print("请输入积分区间 [a, b]，其中 a > 0:")
    a = float(input("a = "))
    b = float(input("b = "))
    if a <= 0:
        raise ValueError("区间起点 a 必须大于 0。")
    
    h = float(input("请输入梯形求积的步长 h: "))
    precision = float(input("请输入精度 E (如 1e-5): "))

    # 创建 NumericalIntegration 对象
    integrator = NumericalIntegration(a=a, b=b, h=h, precision=precision)

    # 比较两种积分方法
    integrator.compare_methods()
    
    # 绘制被积函数
    integrator.plot_function()

if __name__ == "__main__":
    interactive_input()
