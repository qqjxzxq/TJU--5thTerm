import numpy as np
import matplotlib.pyplot as plt

X = input("x的值：").split(' ')
Y = input("y的值：").split(' ')
x = input("要预测的值：")
print('\n')

X = np.array(X).astype(np.float64)
Y = np.array(Y).astype(np.float64)
x = np.array(x).astype(np.float64)

n = len(X)#输入点个数

def fun(x):
    return np.sqrt(x)

def newton(x,y,u):
#差商计算
    c=np.zeros((n,n))
    for i in range(n):
        c[i,0]=y[i]
    for i in range(1,n):
        for j in range(i,n):
            c[j,i]=(c[j,i-1]-c[j-1,i-1])/(x[j]-x[j-i])
#预测值s
    s=y[0]
    for i in range(1,n):
        t=1
        for j in range(0,i):
            t*=(u-x[j])
        s+=c[i,i]*t
    
    return s

y = newton(X, Y, x)

print("预测结果：" + str(y) + '\n')
print("误差：" + str(fun(x) - y))

# 画图
x_n = np.linspace(0, 40, 41)
y_n = fun(x_n)

X_n = np.linspace(0, 40, 41)
Y_n = newton(X, Y, X_n)

l1, = plt.plot(x_n, y_n)
l2, = plt.plot(X_n, Y_n)
plt.scatter(X, Y, color='red', marker='.')
plt.legend(handles=[l1, l2, ], labels=['f(x)', 'g(x)'], loc='best')

plt.show()



    
