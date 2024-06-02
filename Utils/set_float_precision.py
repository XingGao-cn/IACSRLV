from decimal import Decimal, getcontext
import numpy as np


def set_prec(number, number2):
    getcontext().prec = number  # 可修改
    np.set_printoptions(precision=number2,suppress=True)
    # print(getcontext().prec)
