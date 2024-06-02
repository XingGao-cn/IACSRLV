import numpy as np


def degree2rad(degree):
    return np.float64(degree * np.pi / 180.0)


def rad2degree(rad):
    return np.float64(rad * 180.0 / np.pi)

