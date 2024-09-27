## 导入相关包
import numpy as np
import math  # 绝对值函数
from scipy import integrate  # 求积分


## 求梯形值(返回用k阶复化梯形公式估计的积分)
def Trap(f, a, b, Iold, k):
    if k == 1:
        Inew = (f(a) + f(b)) * (b - a) / 2
        print('二分' + str(k - 1) + '次后的梯形值为' + '%.6f' % Inew)
    else:
        n = pow(2, k - 2)
        h = (b - a) / n  # 步长
        x = a + (h / 2)  # 第一步的中心点
        sum_k = 0
        for i in range(n):
            sum_k = sum_k + f(x)  # 求和
            x = x + h  # 下一个点
        Inew = (Iold + h * sum_k) / 2  # 递推公式
        print('二分' + str(k - 1) + '次后的梯形值为' + '%.6f' % Inew)
    return Inew


## 求加速值(运用理查森外推加速算法)
def Richardson(R, k):
    for i in range(k - 1, 0, -1):
        c = pow(2, 2 * (k - i))
        R[i] = (c * R[i + 1] - R[i]) / (c - 1)  # 龙贝格求积算法
    for a in sorted(R.keys(), reverse=True)[1:]:  # 逆序输出
        print('第' + str(k - 1) + '次二分的第' + str(k - a) + '次加速值为' + '%.6f' % R[a])
    return R


## 龙贝格求积分
def romberg(f, a, b, eps):
    T = {}  # 定义空字典
    k = 1
    print('区间[a,b]的二分次数为：' + str(k - 1))
    T[1] = Trap(f, a, b, 0.0, 1)
    former_R = T[1]
    while True:
        k += 1
        print('\n区间[a,b]的二分次数为：' + str(k - 1))
        # 求梯形值
        T[k] = Trap(f, a, b, T[k - 1], k)
        # 求加速值
        T = Richardson(T, k)
        # 判断是否满足终止条件
        if abs(T[1] - former_R) < eps:
            return T[1]
        former_R = T[1]  # 最后一个值置为初始值


## 定义函数
def f(x):
    return x ** (3 / 2)


## 给定参数
a = 0  # 积分上限
b = 1  # 积分下限
eps = 10 ** -5  # 给定精度

## 龙贝格求积分值
I = romberg(f, a, b, eps)
print("\n龙贝格求积分结果为: {:.6f}".format(I))

## 计算机参考值
I_exact, Error = integrate.quad(f, a, b)
print("计算机参考值: {:.6f}".format(I_exact))

print("相对误差(与计算机参考值相比): {:.6f}%".format((I - I_exact) / I_exact * 100))