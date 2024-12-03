import math

def fixed_point_iteration(g, x0, tol=1e-8, max_iter=1000):
    """不动点迭代"""
    x = x0
    for i in range(max_iter):
        x_new = g(x)
        print(f"Fixed-point Iteration {i + 1}: x = {x_new}")
        if abs(x_new - x) < tol:
            return x_new, i + 1
        x = x_new
    raise Exception("Fixed-point iteration did not converge")

def steffensen_acceleration(g, x0, tol=1e-8, max_iter=100):
    """斯特芬森加速迭代"""
    x = x0
    for i in range(max_iter):
        gx = g(x)
        ggx = g(gx)
        if ggx - 2 * gx + x == 0:  # 防止除以零
            raise Exception("Division by zero in Steffensen acceleration")
        x_new = x - (gx - x)**2 / (ggx - 2 * gx + x)
        print(f"Steffensen Acceleration {i + 1}: x = {x_new}")
        if abs(x_new - x) < tol:
            return x_new, i + 1
        x = x_new
    raise Exception("Steffensen acceleration did not converge")

def newton_iteration(f, f_prime, x0, tol=1e-8, max_iter=100):
    """牛顿迭代"""
    x = x0
    for i in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)
        if fpx == 0:  # 防止除以零
            raise Exception("Derivative is zero in Newton iteration")
        x_new = x - fx / fpx
        print(f"Newton Iteration {i + 1}: x = {x_new}")
        if abs(x_new - x) < tol:
            return x_new, i + 1
        x = x_new
    raise Exception("Newton iteration did not converge")

# 求解函数 f(x) = x^2 - 3x + 2 - e^x 的根
def f1(x):
    return x**2 - 3*x + 2 - math.exp(x)

def f1_prime(x):
    return 2*x - 3 - math.exp(x)

def g1(x):
    return (x**2 - math.exp(x) + 2) / 3

# 求解函数 g(x) = x^3 + x^2 + 10x - 20 的根
def f2(x):
    return x**3 + x**2 + 10*x - 20

def f2_prime(x):
    return 3*x**2 + 2*x + 10

def g2(x):
    return (20 - x**3 - x**2)/10

# 主程序
if __name__ == "__main__":
    print("Solving f(x) = x^2 - 3x + 2 - e^x")
    x0 = 0.5
    print("Fixed-point iteration:")
    root1_fp, iter1_fp = fixed_point_iteration(g1, x0)
    print("Steffensen acceleration:")
    root1_steff, iter1_steff = steffensen_acceleration(g1, x0)
    print("Newton iteration:")
    root1_newton, iter1_newton = newton_iteration(f1, f1_prime, x0)

    print("\nSolving g(x) = x^3 + x^2 + 10x - 20")
    x0 = 1.5
    print("Fixed-point iteration:")
    root2_fp, iter2_fp = fixed_point_iteration(g2, x0)
    print("Steffensen acceleration:")
    root2_steff, iter2_steff = steffensen_acceleration(g2, x0)
    print("Newton iteration:")
    root2_newton, iter2_newton = newton_iteration(f2, f2_prime, x0)

    print("\nComparison of methods:")
    print(f"f(x): Fixed-point (iter={iter1_fp}), Steffensen (iter={iter1_steff}), Newton (iter={iter1_newton})")
    print(f"g(x): Fixed-point (iter={iter2_fp}), Steffensen (iter={iter2_steff}), Newton (iter={iter2_newton})")
