# 仿真参数
import numpy as np
from math import pi
from VehSimuParams.atmosphere_model import AtmosphereModel


class SimulationParameters:
    def __init__(self):
        # 起始时刻（s）
        self.t0 = 0
        # 终端时刻（s）
        self.tf = 150
        # 仿真积分步长（s）
        self.dt = 0.001
        # 基础控制器
        self.controller = ""
        # 强化学习控制器
        self.rl_controller = ""
        # 观测器
        self.OBS = ""

    # name is a string variable
    def set_controller(self, name):
        self.controller = name

    def set_rl_controller(self, name):
        self.rl_controller = name

    def set_OBS(self, name):
        self.OBS = name


class InitialState:
    def __init__(self):
        # 初始高度（m）
        self.alt0 = 50000
        # 初始纬度（rad）
        self.phi0 = 60 * pi / 180.0
        # 初始经度（rad）
        self.theta0 = 15 * pi / 180.0
        # 初始速度（Ma）
        self.Ma0 = 15.0
        # 初始声速（m/s）  通过大气模型动态获取初始声速
        _, _, _, self.a0, _ = AtmosphereModel().get_atmosphere_params(self.alt0)

        # 初始速度（m/s）
        self.V0 = self.Ma0 * self.a0

        # 初始航迹角（rad）
        self.gamma0 = 0 * pi / 180.0
        # 初始航向角（rad）
        self.chi0 = 55 * pi / 180.0
        # 初始俯仰角（rad）
        self.pitch0 = 10 * pi / 180.0
        # 初始偏航角（rad）
        self.yaw0 = 0 * pi / 180.0
        # 初始滚转角（rad）
        self.roll0 = 0 * pi / 180.0

        # 初始攻角（rad）
        self.aoa0 = 12 * pi / 180.0
        # 初始侧滑角（rad）
        self.beta0 = 1 * pi / 180.0
        # 初始倾侧角（rad）
        self.sigma0 = 0 * pi / 180.0
        # 初始滚转角速率（rad/s）
        self.p0 = 0 * pi / 180.0
        # 初始俯仰角速率（rad/s）
        self.q0 = 0 * pi / 180.0
        # 初始偏航角速率（rad/s）
        self.r0 = 0 * pi / 180.0


# ine = InitialState()
# print(ine.aoa0)
