import numpy as np
import math
from VehSimuParams.atmosphere_model import AtmosphereModel
from VehSimuParams.environment_params import Constants
# from VehSimuParams.vehicle_params import RLVConstants
from Dynamics.get_aero_coefficients import Aerodynamics


# 六自由度方程
class AttitudeEquation(object):
    def __init__(self):
        self.state = np.zeros((1, 12))
        self.time = np.float64(0)
        self.ENVO_CONST = Constants()  # 物理环境参数常量
        self.ATM_CONST = AtmosphereModel()  # 大气模型常量

    def set_state(self, state):
        self.state = np.reshape(state, (1, -1))

    def get_state(self):
        return self.state

    def set_time(self, time):
        self.time = time

    def get_time(self):
        return self.time


class RLVAttitudeEquation(AttitudeEquation):
    def __init__(self):
        super(RLVAttitudeEquation, self).__init__()  # 初始化父类以继承属性
        self.aero_dynamics = Aerodynamics()
        self.VH_CONST = self.aero_dynamics.VH_CONST  # 飞行器参数常量, 减少类实例

    def get_state_diff(self, time, state, u_degree):
        self.set_state(state)
        self.set_time(time)

        g = 9.80665 / (1 + self.state[0, 0] / (6.356766 * 10 ** 6)) ** 2  # 引力加速度，单位：m / s ^ 2

        rho = self.ENVO_CONST.rho0 * math.exp(-self.state[0, 0] * self.ENVO_CONST.hs)  # 大气密度（kg / m ^ 3）
        CL, CD, CY, Cl, Cm, Cn = (
            self.aero_dynamics.updated_aero_coef(self.state, u_degree))  # 获取气动参数（Cl - 滚转力矩系数, CM - 俯仰力矩系数, CN - 偏航力矩系数）
        q_bar = 0.5 * rho * self.state[0, 3] ** 2  # 动压（Pa）

        L = CL * (q_bar * self.VH_CONST.S_ref)  # 升力
        D = CD * (q_bar * self.VH_CONST.S_ref)  # 阻力
        # Y = CY * (q_bar * self.VH_CONST.S_ref)  # 侧向力
        Mx = q_bar * self.VH_CONST.S_ref * self.VH_CONST.b_bar * Cl  # 滚转力矩（N * m）
        My = q_bar * self.VH_CONST.S_ref * self.VH_CONST.c_bar * Cm  # 俯仰力矩（N * m）
        Mz = q_bar * self.VH_CONST.S_ref * self.VH_CONST.b_bar * Cn  # 偏航力矩（N * m）

        # 加入扰动系数后的力矩与转动惯量
        Mx = (1 + self.VH_CONST.Mx_dis) * Mx  # 滚转力矩（N * m）
        My = (1 + self.VH_CONST.My_dis) * My  # 俯仰力矩（N * m）
        Mz = (1 + self.VH_CONST.Mz_dis) * Mz  # 偏航力矩（N * m）
        Ixx = (1 + self.VH_CONST.Ixx_dis) * self.VH_CONST.Ixx  # x轴转动惯量（kg * m ^ 2）
        Iyy = (1 + self.VH_CONST.Iyy_dis) * self.VH_CONST.Iyy  # y轴转动惯量（kg * m ^ 2）
        Izz = (1 + self.VH_CONST.Izz_dis) * self.VH_CONST.Izz  # z轴转动惯量（kg * m ^ 2）
        Ixz = (1 + self.VH_CONST.Ixz_dis) * self.VH_CONST.Ixz  # x轴和z轴耦合惯性积（kg * m ^ 2）

        # 添加外干扰力矩
        Dx = self.VH_CONST.Dx_1 * (1 + math.sin(math.pi * self.time / self.VH_CONST.Dx_2) +
                                   math.sin(math.pi * self.time / self.VH_CONST.Dx_3))  # 滚转外干扰力矩（N * m）
        Dy = self.VH_CONST.Dy_1 * (1 + math.sin(math.pi * self.time / self.VH_CONST.Dy_2) +
                                   math.sin(math.pi * self.time / self.VH_CONST.Dy_3))  # 俯仰外干扰力矩（N * m）
        Dz = self.VH_CONST.Dz_1 * (1 + math.sin(math.pi * self.time / self.VH_CONST.Dz_2) +
                                   math.sin(math.pi * self.time / self.VH_CONST.Dz_3))  # 偏航外干扰力矩（N * m）

        # 真实作用力矩
        Mx = Mx + Dx
        My = My + Dy
        Mz = Mz + Dz

        # 六自由度十二状态方程
        # [0]高度（m）[1]纬度（rad）[2]经度（rad）[3]速度（m/s）[4]航迹角（rad）[5]航向角（rad）
        # [6]攻角（rad）[7]侧滑角（rad）[8]倾侧角（rad）[9]滚转角速率（rad/s）[10]俯仰角速率（rad/s）[11]偏航角速率（rad/s）
        alt_dot = self.state[0, 3] * math.sin(self.state[0, 4])
        phi_dot = self.state[0, 3] * math.cos(self.state[0, 4]) * math.sin(self.state[0, 5]) / (
                (self.ENVO_CONST.R0 + self.state[0, 0]) * math.cos(self.state[0, 2]))
        theta_dot = self.state[0, 3] * math.cos(self.state[0, 4]) * math.cos(self.state[0, 5]) / (
                self.ENVO_CONST.R0 + self.state[0, 0])
        v_dot = self.ENVO_CONST.omega_e ** 2 * (self.ENVO_CONST.R0 + self.state[0, 0]) * math.cos(self.state[0, 2]) * (
                math.sin(self.state[0, 4]) * math.cos(self.state[0, 2]) - math.cos(self.state[0, 4]) * math.sin(
            self.state[0, 2]) * math.cos(self.state[0, 5])) - D / self.VH_CONST.mass - g * math.sin(self.state[0, 4])
        gamma_dot = L * math.cos(self.state[0, 8]) / (self.VH_CONST.mass * self.state[0, 3]) - (
                g / self.state[0, 3] - self.state[0, 3] / (self.ENVO_CONST.R0 + self.state[0, 0])) * math.cos(
            self.state[0, 4]) + 2 * self.ENVO_CONST.omega_e * math.cos(self.state[0, 2]) * math.sin(
            self.state[0, 5]) + \
                    self.ENVO_CONST.omega_e ** 2 * (self.ENVO_CONST.R0 + self.state[0, 0]) * math.cos(
            self.state[0, 2]) * (
                            math.cos(self.state[0, 4]) * math.cos(self.state[0, 2]) + math.sin(
                        self.state[0, 4]) * math.sin(
                        self.state[0, 2]) * math.cos(self.state[0, 5])) / self.state[0, 3]

        chi_dot = (L * math.sin(self.state[0, 8]) / (
                self.VH_CONST.mass * self.state[0, 3] * math.cos(self.state[0, 4])) +
                   self.state[0, 3] * math.cos(self.state[0, 4]) * math.sin(self.state[0, 5]) * math.tan(
                    self.state[0, 2]) / (
                           self.ENVO_CONST.R0 + self.state[0, 0]) -
                   2 * self.ENVO_CONST.omega_e * (
                           math.tan(self.state[0, 4]) * math.cos(self.state[0, 2]) * math.cos(
                       self.state[0, 5]) - math.sin(
                       self.state[0, 2])) +
                   self.ENVO_CONST.omega_e ** 2 * (self.ENVO_CONST.R0 + self.state[0, 0]) * math.sin(
                    self.state[0, 2]) * math.cos(
                    self.state[0, 2]) * math.sin(self.state[0, 5]) / (self.state[0, 3] * math.cos(self.state[0, 4]))
                   )

        # 姿态角 - 姿态角速率方程
        aoa_dot = -self.state[0, 9] * math.cos(self.state[0, 6]) * math.tan(self.state[0, 7]) + self.state[0, 10] - \
                  self.state[0, 11] * math.sin(self.state[0, 6]) * math.tan(self.state[0, 7])
        beta_dot = self.state[0, 9] * math.sin(self.state[0, 6]) - self.state[0, 11] * math.cos(self.state[0, 6])
        sigma_dot = -self.state[0, 9] * math.cos(self.state[0, 6]) * math.cos(self.state[0, 7]) - self.state[
            0, 10] * math.sin(self.state[0, 7]) - self.state[0, 11] * math.sin(self.state[0, 6]) * math.cos(
            self.state[0, 7])
        p_dot = Izz * Mx / (Ixx * Izz - Ixz ** 2) + Ixz * Mz / (Ixx * Izz - Ixz ** 2) + \
                (Ixx - Iyy + Izz) * Ixz * self.state[0, 9] * self.state[0, 10] / (Ixx * Izz - Ixz ** 2) + (
                        (Iyy - Izz) * Izz - Ixz ** 2) * self.state[0, 10] * self.state[0, 11] / (
                        Ixx * Izz - Ixz ** 2)

        q_dot = My / Iyy + Ixz * (self.state[0, 11] ** 2 - self.state[0, 9] ** 2) / Iyy + (Izz - Ixx) * self.state[
            0, 9] * self.state[0, 11] / Iyy
        r_dot = Ixz * Mx / (Ixx * Izz - Ixz ** 2) + Ixx * Mz / (Ixx * Izz - Ixz ** 2) + \
                ((Ixx - Iyy) * Ixx + Ixz ** 2) * self.state[0, 9] * self.state[0, 10] / (Ixx * Izz - Ixz ** 2) + (
                        Iyy - Ixx - Izz) * Ixz * self.state[0, 10] * self.state[0, 11] / (
                        Ixx * Izz - Ixz ** 2)

        d_state = np.array(
            [alt_dot, phi_dot, theta_dot, v_dot, gamma_dot, chi_dot, aoa_dot, beta_dot, sigma_dot, p_dot, q_dot,
             r_dot]).reshape(1, -1)

        return d_state

# t_ = 30
# state_ = np.ones((1, 12))
# u_degree = np.array([[1], [2], [3]])
# RLVeqn = RLVAttitudeEquation()
# print(RLVeqn.get_state_diff(t_, state_, u_degree).shape)
