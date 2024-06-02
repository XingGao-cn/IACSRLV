import numpy as np


def sig(x, u):
    return np.sign(x) * (abs(x) ^ u)


def sign(x):
    return np.abs(x)
