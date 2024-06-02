import numpy as np
import mpmath
import math
from pathlib import Path
import os

FILE = Path(__file__).resolve()

project_dir = os.path.dirname(FILE)

print(project_dir)

# 定义一个需要求逆矩阵的矩阵
a = np.array([[-0.841470984807897, 1.000000000000000, -1.310513411812786], [0.841470984807897, 0, -0.540302305868140],
              [-0.291926581726429, -0.841470984807897, -0.454648713412841
               ]])

# 使用numpy.linalg.solve()函数求解线性方程组
x = np.linalg.solve(a, np.eye(3))
print(x)

# 使用SVD分解计算矩阵的伪逆
pinv_a = np.linalg.pinv(a)

# 增加计算精度
np.set_printoptions(precision=10)

# 输出结果
print("逆矩阵：n", x)
print("伪逆矩阵：n", pinv_a)

# 设置精度为50位小数
mpmath.mp.dps = 63

# 将其他变量赋值给mpf对象
a = mpmath.mpf(1.12345678901234567890)
b = mpmath.mpf(2.23456789012345678901)

# 进行计算
result = a + b

print(result)

# 创建一个 NumPy 数组
matrix = np.array([[1.123456789, 2.23456789], [3.3456789, 4.456789]])

# 将 NumPy 数组中的数据类型转换为 mp.mpf 对象
mp_matrix = np.vectorize(mpmath.mpf)(matrix)

# 打印转换后的矩阵
print(mp_matrix)

dec_lim = 20  # 舵偏限幅
u_degree = np.array([[10], [20], [30]])
if max(abs(u_degree)) >= dec_lim:
    u_degree = u_degree / max(abs(u_degree)) * dec_lim
print(u_degree)