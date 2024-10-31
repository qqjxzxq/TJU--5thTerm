import numpy as np

def gauss_elimination(A, b):
    n = len(b)
    # 创建增广矩阵
    Ab = np.hstack([A, b.reshape(-1, 1)])
    
    # 列主元消元法
    for i in range(n):
        # 找到主元
        max_row = np.argmax(np.abs(Ab[i:n, i])) + i
        # 交换当前行和最大行
        Ab[[i, max_row]] = Ab[[max_row, i]]
        
        # 消元
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j] -= factor * Ab[i]
    
    # 回代求解
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.sum(Ab[i, i + 1:n] * x[i + 1:n])) / Ab[i, i]
    
    return Ab, x

# 测试输入
A = np.array([
    [31, -13, 0, 0, 0, -10, 0, 0, 0],
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



# 执行消元法
Ab, x = gauss_elimination(A, b)

# 格式化矩阵
formatted_Ab = []
for row in Ab:
    formatted_row = [f"{value:.2f}" if value != 0 else "0" for value in row]
    formatted_Ab.append(formatted_row)

# 格式化解向量
formatted_x = [f"{value:.2f}" if value != 0 else "0" for value in x]

# 随机生成 n 阶方阵 A 和向量 b
def random_gauss_elimination(n):
    A_random = np.round(np.random.rand(n, n) * 100 - 50, 2)  # 随机生成 -50 到 50 的矩阵并保留两位小数
    b_random = np.round(np.random.rand(n) * 100 - 50, 2)  # 随机生成 -50 到 50 的向量并保留两位小数
    Ab_random, x_random = gauss_elimination(A_random, b_random)
    
    # 格式化随机生成的增广矩阵
    formatted_Ab_random = []
    for row in Ab_random:
        formatted_row = [f"{value:.2f}" if value != 0 else "0" for value in row]
        formatted_Ab_random.append(formatted_row)
    
    # 格式化随机生成的解向量
    formatted_x_random = [f"{value:.2f}" if value != 0 else "0" for value in x_random]
    
    return formatted_Ab_random, formatted_x_random

# 测试随机生成
n = 25  # 你可以选择更大的 n
Ab_random, x_random = random_gauss_elimination(n)

# 输出格式化的结果
print("消元后的增广矩阵:")
for row in formatted_Ab:
    print("[" + ", ".join("0" if value in ["0.00", "-0.00"] else value for value in row) + "]")
print("方程 Ax = b 的根值 x*:")
print("["+ ", ".join("0" if value in ["0.00", "-0.00"] else value for value in formatted_x) + "]")

print("\n随机生成的增广矩阵:")
for row in Ab_random:
    print("[" + ", ".join("0" if value in ["0.00", "-0.00"] else value for value in row) + "]")
print("随机生成的方程 Ax = b 的根值 x*:")
print("["+ ", ".join("0" if value in ["0.00", "-0.00"] else value for value in x_random) + "]")
