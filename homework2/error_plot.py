import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# 数据
n = [5, 10, 15, 20]
vandermonde = [0.11115742309965457, 0.00046427904284127323, 2.7897346321775762e-08, 5.57551968483061e-11]
lagrange = [0.11115742309972676, 0.00046427904514932254, 2.787634560196678e-08, 1.950112470811005e-11]
newton = [0.11115742309972734, 0.00046427904514906385, 2.787634920887322e-08, 1.952578043695752e-11]
piecewise_linear = [0.20232147400539005, 0.06442477235638683, 0.0253724430364894, 0.015397208480121579]
hermite = [0.19251559071880478, 0.04664820345867034, 0.01407280235182811, 0.00800343632594138]

# 设置条形宽度
bar_width = 0.15

# 设置位置
r1 = np.arange(len(n))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]
r5 = [x + bar_width for x in r4]

# 创建条形图
plt.bar(r1, vandermonde, width=bar_width, label='范德蒙德', color='red')
plt.bar(r2, lagrange, width=bar_width, label='拉格朗日', color='green')
plt.bar(r3, newton, width=bar_width, label='牛顿', color='r')
plt.bar(r4, piecewise_linear, width=bar_width, label='分段线性', color='c')
plt.bar(r5, hermite, width=bar_width, label='Hermite', color='m')

# 在每个柱形上方标注误差（仅一位有效数字），并增加偏移量
for i in range(len(n)):
    plt.text(r1[i], vandermonde[i] + 1e-12, f"{vandermonde[i]:.1e}", ha='center', va='bottom', rotation=45)
    plt.text(r2[i], lagrange[i] + 1e-12, f"{lagrange[i]:.1e}", ha='center', va='bottom', rotation=45)
    plt.text(r3[i], newton[i] + 1e-12, f"{newton[i]:.1e}", ha='center', va='bottom', rotation=45)
    plt.text(r4[i], piecewise_linear[i] + 1e-12, f"{piecewise_linear[i]:.1e}", ha='center', va='bottom', rotation=45)
    plt.text(r5[i], hermite[i] + 1e-12, f"{hermite[i]:.1e}", ha='center', va='bottom', rotation=45)

# 添加标签和标题
plt.xlabel('n', fontweight='bold')
plt.ylabel('均方误差', fontweight='bold')
plt.title('误差条形图', fontweight='bold')
plt.xticks([r + 2 * bar_width for r in range(len(n))], n)  # 设置x轴刻度

# 添加图例
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()
