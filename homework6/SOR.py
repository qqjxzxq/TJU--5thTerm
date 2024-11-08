import numpy as np

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

n = A.shape[0]
err = 1
eps = 1e-8
N = 100  # 最大迭代次数
iter_count = 0
solution = np.zeros(n)

def chaosongchi(A, b, w, eps=1e-8, N=100):
    n = A.shape[0]
    solution = np.zeros(n)
    err = 1
    iter_count = 0
    while err > eps and iter_count < N:
        iter_count += 1
        x0 = solution.copy()
        for i in range(n):
            sum1 = np.dot(A[i, 0:i], solution[0:i])   # A[i, 0:i] * solution[0:i]
            sum2 = np.dot(A[i, i+1:n], x0[i+1:n])  # A[i, i+1:n] * x0[i+1:n]
            solution[i] = (b[i] - sum1 - sum2) / A[i, i]  # 更新解
            solution[i] = (1 - w) * x0[i] + w * solution[i]  # 松弛更新

        err = np.max(np.abs(solution - x0))  # 计算误差

    return solution, iter_count, err

# 选择最佳松弛因子
best_iter = float('inf')
best_w = None
best_solution = None

for i in range(1, 100):  # 松弛因子 i/50, i = 1,2,...,99
    w = i / 50.0
    solution, iter_count, err = chaosongchi(A, b, w, eps=eps, N=N)
    print(f"松弛因子 {w:.2f} 的超松弛迭代法迭代次数：{iter_count}")
    print(f"松弛因子 {w:.2f} 的超松弛迭代法误差：{err}")
    print(f"松弛因子 {w:.2f} 的超松弛迭代法解：{solution}\n")

    # 更新最优松弛因子
    if iter_count < best_iter:
        best_iter = iter_count
        best_w = w
        best_solution = solution

# 打印最佳松弛因子
print(f"最佳松弛因子是 {best_w:.2f}，对应的迭代次数为 {best_iter}，解为：")
print(best_solution)
