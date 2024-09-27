import numpy as np
# e为误差
def Gauss_Seidel(A, b, x, e, times=100):
    D = np.mat(np.diag(np.diag(A)))#去对角线值
    L = np.triu(A, 1)#取下三角值
    DL_inv = np.linalg.inv(D+L)
    U = np.tril(A, -1)#取上三角值
    B = -DL_inv * U
    x0 = x#初始x为0
    x = B * x0 + DL_inv * b
    k = 1
    print('Gauss_Seidel第', k, '次', '迭代解为\n', x)
    while k < times:
        if abs(np.max(abs(x - x0), axis=0)) > e:
            x0 = x
            x = B * x0 + DL_inv * b
            k += 1
            print('Gauss_Seidel第',k,'次','迭代解为\n', x)
        else:
            print('当精度为', e, '时,Gauss_Seidel在%d次内收敛' % k)
            break


A = np.mat([[8, -1, 1], [2, 10, -1], [1, 1, -5]])
b = np.mat('1;4;3')
x = np.mat('0;0;0')
e = 0.001
times = 100
Gauss_Seidel(A, b, x, e, times)