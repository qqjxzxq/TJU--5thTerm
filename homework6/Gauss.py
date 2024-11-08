import numpy as np
import math
import sys
#分解矩阵
def DLU(A):
    D=np.zeros(np.shape(A))
    L=np.zeros(np.shape(A))
    U=np.zeros(np.shape(A))
    for i in range(A.shape[0]):
        D[i,i]=A[i,i]
        for j in range(i):
            L[i,j]=-A[i,j]
        for k in list(range(i+1,A.shape[1])):
            U[i,k]=-A[i,k]
    L=np.mat(L)
    D=np.mat(D)
    U=np.mat(U)
    return D,L,U

#迭代
def Jacobi_iterative(A,b,x0,maxN,p):  #x0为初始值，maxN为最大迭代次数，p为允许误差
    D,L,U=DLU(A)
    if len(A)==len(b):
        M=np.linalg.inv(D-L)
        D_inv=np.mat(M)
        B=D_inv * U
        B=np.mat(B)
        f=M*b
        f=np.mat(f)
    else:
        print('维数不一致')
        sys.exit(0)  # 强制退出
    
    a,b=np.linalg.eig(B) #a为特征值集合，b为特征值向量
    c=np.max(np.abs(a)) #返回谱半径
    if c<1:
        print('迭代收敛')
    else:
        print('迭代不收敛')
        sys.exit(0)  # 强制退出
#以上都是判断迭代能否进行的过程，下面正式迭代
    k=0
    while k<maxN:
        x=B*x0+f
        k=k+1
        eps=np.linalg.norm(x-x0,ord=2)
        if eps<p:
            break
        else:
            x0=x
    return k,x

A = np.array([
    [-31, 13, 0, 0, 0, -10, 0, 0, 0],
    [-13, 35, -9, 0, -11, 0, 0, 0, 0],
    [0, -9, 31, -10, 0, 0, 0, 0, 0],
    [0, 0, -10, 79, -30, 0, 0, 0, -9],
    [0, 0, 0, -30, 57, -7, 0, -5, 0],
    [0, 0, 0, 0, -7, 47, -30, 0, 0],
    [0, 0, 0, 0, 0, -30, 41, 0, 0],
    [0, 0, 0, 0, -5, 0, 0, 27, -2],
    [0, 0, 0, -9, 0, 0, 0, -2, 29]
], dtype=float)

b = np.array([-15, 27, -23, 0, -20, 12, -7, 7, 10], dtype=float)
b = np.mat(b).T

x0=np.mat([[0],[0],[0],[0],[0],[0],[0],[0],[0]])
maxN=100
p=1e-8
print("原系数矩阵a:")
print(A, "\n")
print("原值矩阵b:")
print(b, "\n")
k,x=Jacobi_iterative(A,b,x0,maxN,p)
print("迭代次数")
print(k, "\n")
print("数值解")
print(x, "\n")