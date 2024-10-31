import numpy as np

def LuFactorization(A):
    n = len(A)
    
    L = np.eye(n, dtype=float)  # 初始化 L 矩阵为单位矩阵
    U = np.zeros((n, n), dtype=float)  # 初始化 U 矩阵为零矩阵

    for i in range(n):
        # 计算 U 的第 i 行
        U[i, i:] = A[i, i:] - L[i, :i] @ U[:i, i:]
        
        # 计算 L 的第 i 列
        if i < n - 1:
            L[i + 1:, i] = (A[i + 1:, i] - L[i + 1:, :i] @ U[:i, i]) / U[i, i]
    
    return L, U

def BackSubstitution(M, b, reverse=False):
    n = len(b)
    x = np.zeros(n, dtype=float)
    index = list(range(n))
    
    if reverse:
        index.reverse()
        
    for k in index:
        x[k] = (b[k] - M[k] @ x) / M[k][k]
        x[k] = round(x[k], 2)
    
    
    return x

def CalcLinearEquation(A, b, fac_type=0):
    # 使用 LU 分解
    if fac_type == 0:
        L, U = LuFactorization(A)
        print("L 矩阵:")
        print(np.round(L, 2))
        print("\nU 矩阵:")
        print(np.round(U, 2))
        return BackSubstitution(U, BackSubstitution(L, b), reverse=True)
    return None

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

b = np.array([188, -3145, -4994, 680, 7845, 1876, 9712, -11599, 10127], dtype=float)

# 求解
result = CalcLinearEquation(A, b)
result_rounded = np.round(result, 2)
print("\n方程 Ax = b 的解 x*:")
np.set_printoptions(precision=2, suppress=True)  # 设置输出格式
print(result_rounded)


# 随机生成 n 阶方阵 A 和非零向量 b
n = 25  # 你可以修改这个值
A = np.random.randint(-100, 100, size=(n, n)).astype(float)  # 随机生成 n x n 方阵
b = np.random.randint(-100, 100, size=(n,)).astype(float)  # 随机生成 n 维非零向量

# 确保 b 不全为零
while np.all(b == 0):
    b = np.random.randint(-100, 100, size=(n,)).astype(float)

# 求解
print("\n随机生成的矩阵分解成L、U矩阵： ")
result0 = CalcLinearEquation(A, b)

# 输出结果
print("\n随机生成 方程 Ax1 = b 的解 x1*:")
print(np.round(result0, 2))


