import numpy as np
from TrackingDifferentiator.ADRCTD import fst_func


# 在 Matlab代码中使用的是persistent关键字，在此使用类变量替代函数内部静态变量
# 在一次迭代完成后，需要重新实例化TDIterator类，或者覆盖self.init
class TDIterator:
    # 初始的 v
    def __init__(self, v):
        self.x1 = v
        self.x2 = 0

    def TD(self, v, T):
        # v 是原始期望指令, x2 = x1_dot, x1是滤波后的指令, T为积分步长.
        # T = 0.001;  # 积分步长,  韩老师文章记作 h.
        r = 1000  # 决定跟踪快慢的参数.
        r2 = 1
        h = 0.008  # 滤波性能与相位损失之间的权衡参数.
        h1 = 8 * h

        # # 初始化变量
        # if x1 is None:
        #     x1 = v
        # if x2 is None:
        #     x2 = 0

        x1_k = self.x1
        x2_k = self.x2

        # 论文中 TD 的离散化公式.
        # x1(k+1) = x1(k) + h*x2(k)
        # x2(k+1) = x2(k) + h*fst(x1(k)-v(k), x2(k), r, h1)
        self.x1 = x1_k + T * x2_k
        self.x2 = x2_k + T * fst_func.fst(v, x1_k, x2_k, r, h1, r2)  # 因为x1k是逐步算出的，故单独传入x1(k)与v(k)

        # 输出
        x1_ = self.x1
        x2_ = self.x2

        return x1_, x2_


# 示例用法

# v_example = 150.0  # 请根据需要设置具体的v值
# T_example = 0.01  # 请根据需要设置具体的T值
# TD_instance = TDIterator(v_example)
# result1, result2 = TD_instance.TD(v_example, T_example)
# print(result1, result2)
