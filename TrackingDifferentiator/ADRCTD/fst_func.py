# 最优最速函数 供td.py调用

import numpy as np


def fst(v, x1_k, x2_k, r, h1, r2):
    # fst = −r * sat(g(k), delta), 快速控制最优综合函数, 也称fhan函数
    delta = h1 * r  # h1为滤波因子  r为调节系数，r越大跟踪效果越好，但微分信号会增加高频噪声
    delta1 = delta * h1  # 反之，微分信号越平滑，会产生一定的滞后
    e_k = x1_k - v
    y_k = e_k + r2 * h1 * x2_k

    g_k = g(x2_k, y_k, delta, delta1, r, h1)
    f = -r * sat(g_k, delta)

    return f


def g(x2_k, y_k, delta, delta1, r, h1):
    a0 = np.sqrt(delta ** 2 + 8 * r * np.abs(y_k))

    if np.abs(y_k) >= delta1:
        g_k = x2_k + (a0 - delta) * np.sign(y_k) / 2
    else:
        g_k = x2_k + y_k / h1

    return g_k


def sat(x, delta):
    if np.abs(x) >= delta:
        f = np.sign(x)
    else:
        f = x / delta

    return f

# # 示例用法
# v_example = 1.0  # 请根据需要设置具体的v值
# x1_k_example = 2.0  # 请根据需要设置具体的x1_k值
# x2_k_example = 3.0  # 请根据需要设置具体的x2_k值
# r_example = 0.5  # 请根据需要设置具体的r值
# h1_example = 0.1  # 请根据需要设置具体的h1值
# r2_example = 0.2  # 请根据需要设置具体的r2值
#
# result = fst(v_example, x1_k_example, x2_k_example, r_example, h1_example, r2_example)
# print(result)
