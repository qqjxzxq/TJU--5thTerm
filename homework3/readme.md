第三章       函数拟合

任务一：改善龙格现象
    源代码：Runge.py 
    运行： python Runge.py
        输入插值区间：-1 1
        输入采样个数 n ：n
        输入实验点个数 m ：m 
    #使用随机采样：注释掉28行，打开29行
    #使用切比雪夫采样：注释掉29行，打开28行

任务二：最小二乘法函数拟合
2.1
    源代码：CurveFitting1.py
    批量脚本测试函数：test_curve_fitting1.py
    运行：
        1.不使用批量测试脚本：python CurveFitting1.py  ,然后根据命令行提示输入即可
        2.批量测试：python test_curve_fitting1.py ,测试数据在函数内部传入