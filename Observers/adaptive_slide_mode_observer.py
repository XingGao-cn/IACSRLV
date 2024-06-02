import math
import numpy as np
from Controllers.set_controller import Controller
from Observers.set_observer import Observers
from Dynamics.get_aero_coefficients import Aerodynamics
from VehSimuParams.simulation_params import SimulationParameters
from Utils import set_float_precision
from Utils.degree_rad_transformation import rad2degree


# ********************观测器实例*************************
# 自适应多变量滑模扰动观测器

class AMSDO(Observers):
    def __init__(self, state_init, command_init, obs_params_init):
        super(AMSDO, self).__init__("Adaptive Sliding Mode Disturbance Observer",
                                    state_init, command_init, obs_params_init)  # 初始化父类以继承属性

        self.aero_dynamics = Aerodynamics()  # 用于更新气动参数
        # 观测器参数作为子类变量,方便调试
        self.lambda0 = 5
        self.lambda1 = 6
        self.lambda2 = 10
        self.lambda3 = 1
        self.delta1 = 0.1

    # 根据观测器特性获取扰动观测值
    def get_disturbances_obs(self, state, obs_params):
        self.set_state(state)
        self.set_obs_params(obs_params)

        rho = self.ENVO_CONST.rho0 * math.exp(-self.state[0, 0] * self.ENVO_CONST.hs)
        q_bar = 0.5 * rho * self.state[0, 3] ** 2
        # _, _, _, Cl, Cm, Cn = (
        #     self.aero_dynamics.updated_aero_coef(self.state, u_degree))  # 获取气动参数（Cl - 滚转力矩系数, CM - 俯仰力矩系数, CN - 偏航力矩系数）
        #
        # Mx = q_bar * self.VH_CONST.S_ref * self.VH_CONST.b_bar * Cl  # 滚转力矩（N * m）
        # My = q_bar * self.VH_CONST.S_ref * self.VH_CONST.c_bar * Cm  # 俯仰力矩（N * m）
        # Mz = q_bar * self.VH_CONST.S_ref * self.VH_CONST.b_bar * Cn  # 偏航力矩（N * m）

        R = np.array([[-math.cos(self.state[0, 6]) * math.tan(self.state[0, 7]), 1,
                       -math.sin(self.state[0, 6]) * math.tan(self.state[0, 7])],
                      [math.sin(self.state[0, 6]), 0, -math.cos(self.state[0, 6])],
                      [-math.cos(self.state[0, 6]) * math.cos(self.state[0, 7]), -math.sin(self.state[0, 7]),
                       -math.sin(self.state[0, 6]) * math.cos(self.state[0, 7])]])

        w = np.array([[self.state[0, 9]], [self.state[0, 10]], [self.state[0, 11]]])

        # I = np.array([[self.VH_CONST.Ixx, 0, -self.VH_CONST.Ixz],
        #               [0, self.VH_CONST.Iyy, 0],
        #               [-self.VH_CONST.Ixz, 0, self.VH_CONST.Izz]])
        #
        # Omega = np.array([[0, -self.state[0, 11], self.state[0, 10]],
        #                   [self.state[0, 11], 0, -self.state[0, 9]],
        #                   [-self.state[0, 10], self.state[0, 9], 0]])

        aero_angle_d_dot = self.command[0, 3:6].reshape(-1, 1)  # 一阶导

        xi = self.obs_params[0, 4:7].reshape(-1, 1)  # 形状与e2保持一致
        e2 = R @ w - aero_angle_d_dot
        D = xi + np.multiply(self.lambda2, e2)

        self.set_D(D)  # 更新自身类变量

        return D

    # 根据观测器中间参数微分方程
    def get_params_diff(self, time, obs_params, state, command, u_degree, D):
        self.set_state(state)
        self.set_command(command)
        self.set_obs_params(obs_params)
        self.set_D(D)

        # print("滑模异常")
        # print(self.state[0, 0])
        # print(self.ENVO_CONST.hs)
        rho = self.ENVO_CONST.rho0 * math.exp(-self.state[0, 0] * self.ENVO_CONST.hs)

        # print("rho")
        # print(rho)

        q_bar = 0.5 * rho * self.state[0, 3] ** 2
        _, _, _, Cl, Cm, Cn = (
            self.aero_dynamics.updated_aero_coef(self.state, u_degree))  # 获取气动参数（Cl - 滚转力矩系数, CM - 俯仰力矩系数, CN - 偏航力矩系数）

        # print("C")
        # print(Cl)
        # print(Cm)
        # print(Cn)

        Mx = q_bar * self.VH_CONST.S_ref * self.VH_CONST.b_bar * Cl  # 滚转力矩（N * m）
        My = q_bar * self.VH_CONST.S_ref * self.VH_CONST.c_bar * Cm  # 俯仰力矩（N * m）
        Mz = q_bar * self.VH_CONST.S_ref * self.VH_CONST.b_bar * Cn  # 偏航力矩（N * m）

        # print("M")
        # print(Mx)
        # print(My)
        # print(Mz)

        R = np.array([[-math.cos(self.state[0, 6]) * math.tan(self.state[0, 7]), 1,
                       -math.sin(self.state[0, 6]) * math.tan(self.state[0, 7])],
                      [math.sin(self.state[0, 6]), 0, -math.cos(self.state[0, 6])],
                      [-math.cos(self.state[0, 6]) * math.cos(self.state[0, 7]), -math.sin(self.state[0, 7]),
                       -math.sin(self.state[0, 6]) * math.cos(self.state[0, 7])]])

        w = np.array([[self.state[0, 9]], [self.state[0, 10]], [self.state[0, 11]]])

        I = np.array([[self.VH_CONST.Ixx, 0, -self.VH_CONST.Ixz],
                      [0, self.VH_CONST.Iyy, 0],
                      [-self.VH_CONST.Ixz, 0, self.VH_CONST.Izz]])

        Omega = np.array([[0, -self.state[0, 11], self.state[0, 10]],
                          [self.state[0, 11], 0, -self.state[0, 9]],
                          [-self.state[0, 10], self.state[0, 9], 0]])

        Moment = np.array([[Mx], [My], [Mz]])

        aero_angle_d_dot = self.command[0, 3:6].reshape(3, 1)
        aero_angle_2d_dot = self.command[0, 6:9].reshape(3, 1)

        e2 = R @ w - aero_angle_d_dot
        # print("E")
        # print(e2)

        if time == 0:
            s = np.multiply(0.001, np.ones((3, 1)))
        else:
            eta = self.obs_params[0, 1:4].reshape(-1, 1)
            s = e2 - eta

        # print("s")
        # print(s)

        # 计算 upsilon等中间参数
        upsilon = np.multiply(self.lambda1, np.sign(s))

        # print("upsilon")
        # print(upsilon)

        beta1_bar_dot = np.array(
            - self.delta1 * self.obs_params[0, 0] + np.multiply(2, np.linalg.norm(upsilon, ord=2))).reshape((1, 1))

        # print("beta1_bar_dot")
        # print(beta1_bar_dot)

        eta_dot = self.D - R @ np.linalg.inv(I) @ Omega @ I @ w - aero_angle_2d_dot + R @ np.linalg.inv(
            I) @ Moment + np.multiply(self.lambda0, s) + np.multiply(self.lambda1, np.sign(s))

        # print("beta1_bar_dot")
        # print(eta_dot)

        xi_dot = np.multiply(self.lambda2, (
                    -self.D + R @ np.linalg.inv(I) @ Omega @ I @ w + aero_angle_2d_dot - R @ np.linalg.inv(
                I) @ Moment)) + np.multiply(self.obs_params[0, 0] + self.lambda3, np.sign(upsilon))

        # print("xi_dot")
        # print(xi_dot)


        # print("seven")

        # beta1_bar_dot = np.reshape(beta1_bar_dot, (1, -1))
        # eta_dot = np.reshape(eta_dot, (1, -1))
        # xi_dot = np.reshape(xi_dot, (1, -1))

        # print(beta1_bar_dot.shape)
        # print(eta_dot.shape)
        # print(xi_dot.shape)

        # 构建 d_D_param 行向量 (水平拼接)
        # d_param_diff = np.hstack((beta1_bar_dot, eta_dot, xi_dot))
        d_param_diff = np.concatenate((beta1_bar_dot, eta_dot, xi_dot), axis=0).reshape(1, -1)
        # print("d_param_diff")
        # print(d_param_diff.shape)
        # print(d_param_diff.T.shape)

        return d_param_diff


# time_ = 10
# state_init_ = 5*np.ones((1, 12))
# command_init_ = 5*np.ones((1, 9))
# obs_params_init_ = 5*np.ones((1, 7))
# u_degree_ = 5*np.ones((3, 1))
# D_inti = 5*np.ones((3, 1))
# asmdo = AMSDO(state_init_, command_init_, obs_params_init_)
# print(asmdo.get_params_diff(time_, obs_params_init_, state_init_, command_init_, u_degree_, D_inti))

# 参考文献
# Y. Zhu, J. Qiao and L. Guo, "Adaptive Sliding Mode Disturbance Observer-Based Composite Control With Prescribed
# Performance of Space Manipulators for Target Capturing," in IEEE Transactions on Industrial Electronics, vol. 66,
# no. 3, pp. 1973-1983, March 2019, doi: 10.1109/TIE.2018.2838065.
