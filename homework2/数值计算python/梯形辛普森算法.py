import math
import numpy as np
import matplotlib.pyplot as plt
#待求解数值积分sqrt(x) * log(x)
def f1(x):
    if (float(np.fabs(x))<1e-15) :
        return 0
    y=np.sqrt(x) * np.log(x)
    return y
#待求解数值积分sin(x)/x
def f2(x):
    if (float(np.fabs(x)) < 1e-15):
        return 1
    y=np.sin(x)/x
    return y
#梯形公式 f为待求解积分 a为积分下限 b为积分上限
def TX(f,a,b):
    TX = 0.5 * (b - a) * (f(a) + f(b))
    print("梯形公式计算结果为：TX = ", TX)
#辛普森公式 f为待求解积分 a为积分下限 b为积分上限
def XPS(f,a,b):
    XPS = (b-a)*(f(a)+4*f((a+b)/2)+f(b))/6.0
    print("辛普森公式计算结果为：XPS = ", XPS)
#复化梯形公式 f为待求解积分 a为积分下限 b为积分上限 n为区间等分数
def FHTx(f,a,b,n):
    ti=0.0
    h=(b-a)/n
    ti=f(a)+f(b)
    for k in range(1,int(n)):
        xk=a+k*h
        ti = ti + 2 * f(xk)
    FHTx = ti*h/2
    return FHTx
#复化辛普森公式 f为待求解积分 a为积分下限 b为积分上限 n为区间等分数
def FHXPs(f,a,b,n):
    si=0.0
    h = (b - a) / (2 * n)
    si=f(a)+f(b)
    for k in range(1,int(n)):
        xk = a + k * 2 * h
        si = si + 2 * f(xk)
    for k in range(int(n)):
        xk = a + (k * 2 + 1) * h
        si = si + 4 * f(xk)
    FHXPs = si*h/3
    return FHXPs
def FHTx_Rf(a,b,e):
    k = 1
    Tn2 = FHTx(f2, a, b, 2*k)
    Tn1 = FHTx(f2, a, b, k)
    FHTx_er = abs(Tn2 - Tn1)
    while(FHTx_er >= 3*e):
        k = k + 1
        Tn2 = FHTx(f2, a, b, 2 * k)
        Tn1 = FHTx(f2, a, b, k)
        FHTx_er = abs(Tn2 - Tn1)
    print("当n的值为：",2*k,"梯形公式事后估计误差为：",FHTx_er,"满足误差条件e为：",e)

def FHXPs_Rf(a,b,e):
    t = 1
    Tn2 = FHXPs(f2, a, b, 2* t)
    Tn1 = FHXPs(f2, a, b, t)
    FHXPs_er = abs(Tn2 - Tn1)
    while(FHXPs_er >= 15*e):
        t = t + 1
        Tn2 = FHXPs(f2, a, b, 2 * t)
        Tn1 = FHXPs(f2, a, b, t)
        FHXPs_er = abs(Tn2 - Tn1)
    print("当n的值为：", 2*t, "辛普森公式事后估计误差为：", FHXPs_er, "满足误差条件e为：", e)
if __name__ == '__main__':
    a = input("积分下限a = ")  # 积分下限
    b = input("积分上限b = ")  # 积分上限
    e = input("误差精度e = ")  # 误差精度
    a = float(a)  # 强制转换为float类型
    b = float(b)
    e = float(e)
    FHTx_Rf(a, b, e)  # 事后估计梯形求解n
    FHXPs_Rf(a, b, e)  # 事后估计辛普森求解n




