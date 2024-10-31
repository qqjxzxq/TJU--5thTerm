import numpy as np

# 输入矩阵的行列数
input001 = list(map(int, input("请输入矩阵的行列数（例如：3）: ").split()))
n = int(input001[0])

# 输入矩阵 NxN
a = np.zeros((n, n), dtype=np.double)
print("请输入矩阵的每一行元素（以空格分隔）:")
for r in range(n):
    a[r, :] = np.array(input().split(), dtype=np.double)

# 输入常数矩阵 b
b = np.zeros((n, 1), dtype=np.double)
print("输入常数矩阵 b 的每一个元素:")
for r in range(n):
    b[r] = np.array(input(), dtype=np.double)

# LU分解
for i in range(n):
    max_row = i  # 换行标记符
    for i1 in range(i, n):
        # 找到该列元素中的最大值
        if a[max_row][i] < a[i1][i]:
            max_row = i1
    # 若没有交换行元素
    if max_row == i:
        for k in range(i, n - 1):
            x = a[k + 1][i] / a[i][i]
            # 给下三角的一个元素赋值
            a[k + 1][i] = x
            # 计算上三角i+1行的右边值
            for m in range(i + 1, n):
                a[k + 1][m] = a[k + 1][m] - x * a[i][m]
    else:
        # a矩阵交换两行
        for i2 in range(i, n):
            temp = a[i][i2]
            a[i][i2] = a[max_row][i2]
            a[max_row][i2] = temp
        # b矩阵交换两行
        temp = b[max_row][0]
        b[max_row][0] = b[i][0]
        b[i][0] = temp
        # a矩阵交换两行
        for i3 in range(0, i):
            temp = a[i][i3]
            a[i][i3] = a[max_row][i3]
            a[max_row][i3] = temp
        for k in range(i, n - 1):
            x = a[k + 1][i] / a[i][i]
            a[k + 1][i] = x
            for m in range(i + 1, n):
                a[k + 1][m] = a[k + 1][m] - x * a[i][m]

print("LU分解后的矩阵 A:")
print(a)

# 使用向前替代法求解方程
for i in range(1, n):
    x = 0
    for i1 in range(0, i):
        x = a[i][i1] * b[i1][0] + x
    b[i] = (b[i][0] - x)

# 回代法
b[n - 1][0] = b[n - 1][0] / a[n - 1][n - 1]
for i in range(n - 2, -1, -1):
    x = 0
    for i1 in range(i, n - 1):
        x = x + a[i][i1 + 1] * b[i1 + 1][0]
    b[i][0] = (b[i][0] - x) / a[i][i]

print("解 b:")
print(b)
