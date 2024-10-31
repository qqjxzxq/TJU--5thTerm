import numpy as np

# 测试输入
A = np.array([
    [30, 33, -43, -11, -38, -29, 37, 28, 23],
    [-480, -523, 644, 128, 621, 480, -618, -489, -329],
    [60, 266, -1862, -1991, 464, 546, -968, -1567, 1652],
    [540, 624, -782, 290, -893, 123, 567, 5, -122],
    [-450, -675, 2245, 2326, -1512, 1230, -822, 129, -189],
    [-300, -120, -1114, -1295, 1946, 302, -376, -1540, -609],
    [1080, 998, 508, 2460, -1628, -1358, 2896, 2828, -2002],
    [-1080, -1408, 3340, 2267, 21, -1202, 866, -2690, -1351],
    [-300, -435, 1594, 1685, 340, 2279, -27, 2917, -2336]
], dtype=float)

b = np.array([188, -3145, -4994, 680, 7845, 1876, 9712, -11599, 10127], dtype=float).reshape(-1, 1)

n = A.shape[0]

# LU分解
for i in range(n):
    max_row = i
    for k in range(i + 1, n):
        if abs(A[k][i]) > abs(A[max_row][i]):
            max_row = k
    # 交换行
    if max_row != i:
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]

    for k in range(i + 1, n):
        factor = A[k][i] / A[i][i]
        A[k][i] = factor
        for j in range(i + 1, n):
            A[k][j] -= factor * A[i][j]

# 向前替代法求解
for i in range(1, n):
    for j in range(i):
        b[i] -= A[i][j] * b[j]

# 回代法
x = np.zeros((n, 1))
x[-1] = b[-1] / A[-1][-1]
for i in range(n - 2, -1, -1):
    sum_ax = 0
    for j in range(i + 1, n):
        sum_ax += A[i][j] * x[j]
    x[i] = (b[i] - sum_ax) / A[i][i]

# 输出结果
print("解 x:")
print(np.round(x, 2))
