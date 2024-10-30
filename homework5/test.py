import numpy as np

def maximum_elimination(A, B):
    n = len(B)
    A1 = np.hstack((A, B.reshape(-1, 1)))  # 创建增广矩阵

    for i in range(n - 1):  # 遍历每一行
        # 找到第 i 列的主元
        index = np.argmax(np.abs(A1[i:n, i])) + i
        # 交换当前行和最大行
        A1[[i, index]] = A1[[index, i]]

        # 消元过程
        for j in range(i + 1, n):
            a = A1[j, i] / A1[i, i]
            A1[j] -= A1[i] * a

    X = backsub(A1[:, :-1], A1[:, -1])  # 回代求解
    return X

def backsub(A, B):
    if np.linalg.det(A) == 0:  # 检查矩阵是否奇异
        return None
    n = len(B)
    X = np.zeros(n)
    X[-1] = B[-1] / A[-1, -1]  # 最后一行的解
    for k in range(n - 2, -1, -1):
        X[k] = (B[k] - A[k, k + 1:] @ X[k + 1:]) / A[k, k]  # 回代
    return X

# 测试输入
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

B = np.array([-15, 27, -23, 0, -20, 12, -7, 7, 10], dtype=float)

# 执行列主元消元法
X = maximum_elimination(A, B)

# 输出结果
print("方程 Ax = b 的解 x*:")
print(X)
