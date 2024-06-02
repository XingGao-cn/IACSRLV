from TrackingDifferentiator.ADRCTD import command_2

# 绘图所用packets
import numpy as np
from VehSimuParams.simulation_params import SimulationParameters
from TrackingDifferentiator.Filter.runge4kutta_filter import rk4_filter
from matplotlib import pyplot as plt  # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
from pathlib import Path
from math import pi
import os

script_dir = Path(__file__).parent
parent_dir = script_dir.parent.parent


class Filter():
    def __init__(self):
        # 基础类变量 如无特殊设置, 参数遵循SimulationParameters
        self.t0 = SimulationParameters().t0
        self.dt = SimulationParameters().dt
        self.end = SimulationParameters().tf
        self.rad_command = np.zeros((1, 1))
        self.filter_command = np.zeros((1, 1))

        # 滤波器超参
        self.tao = 0.985

    def get_raw_command(self, t0, dt, end):
        # 获取command2中的原始指令
        self.t0 = t0
        self.dt = dt
        self.end = end

        length = np.arange(self.t0, self.end + self.dt, self.dt)
        v = np.zeros((len(length), 3))
        # 获取原始指令
        for i in range(len(length)):
            tmp = command_2.get_command2(length[i], end)
            v[i, :] = tmp[0:3]
        self.rad_command = v

        return self.rad_command

    def get_filter_command(self, t0, dt, end):
        v = self.get_raw_command(t0, dt, end)  # n × 3

        # 通道一(攻角)通过 TD 后的指令以及一阶，二阶导数
        v1 = np.zeros_like(v[:, 0])
        v1_dot = np.zeros_like(v[:, 0])
        v1_d_dot = np.zeros_like(v[:, 0])

        for i in range(len(v)):
            if i == 0:
                v1[i], v1_dot[i] = rk4_filter(Filter().filter, v[i, 0], v[i, 0])
            else:
                v1[i], v1_dot[i] = rk4_filter(Filter().filter, v[i, 0], v1[i - 1])

        for i in range(len(v)):
            if i == 0:
                _, v1_d_dot[i] = rk4_filter(Filter().filter, v1_dot[i], v1_dot[i])
            else:
                _, v1_d_dot[i] = rk4_filter(Filter().filter, v1_dot[i], v1_dot[i - 1])

        # 通道二(侧滑)通过TD后的指令以及一阶，二阶导数
        v2 = np.zeros_like(v[:, 1])
        v2_dot = np.zeros_like(v[:, 1])
        v2_d_dot = np.zeros_like(v[:, 1])

        for i in range(len(v)):
            if i == 0:
                v2[i], v2_dot[i] = rk4_filter(Filter().filter, v[i, 1], v[i, 1])
            else:
                v2[i], v2_dot[i] = rk4_filter(Filter().filter, v[i, 1], v2[i - 1])

        for i in range(len(v)):
            if i == 0:
                _, v2_d_dot[i] = rk4_filter(Filter().filter, v2_dot[i], v2_dot[i])
            else:
                _, v2_d_dot[i] = rk4_filter(Filter().filter, v2_dot[i], v2_dot[i - 1])

        # 通道三(倾侧)通过TD后的指令以及一阶，二阶导数
        v3 = np.zeros_like(v[:, 2])
        v3_dot = np.zeros_like(v[:, 2])
        v3_d_dot = np.zeros_like(v[:, 2])

        for i in range(len(v)):
            if i == 0:
                v3[i], v3_dot[i] = rk4_filter(Filter().filter, v[i, 2], v[i, 2])
            else:
                v3[i], v3_dot[i] = rk4_filter(Filter().filter, v[i, 2], v3[i - 1])

        for i in range(len(v)):
            if i == 0:
                _, v3_d_dot[i] = rk4_filter(Filter().filter, v3_dot[i], v3_dot[i])
            else:
                _, v3_d_dot[i] = rk4_filter(Filter().filter, v3_dot[i], v3_dot[i - 1])

        self.filter_command = np.column_stack((v1, v2, v3, v1_dot, v2_dot, v3_dot, v1_d_dot, v2_d_dot, v3_d_dot))

        return self.filter_command

    def filter(self, x_d, x_f):
        # 滤波器微分方程
        x_dot = (x_d - x_f) / self.tao

        return x_dot
