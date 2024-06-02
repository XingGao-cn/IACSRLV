import math
import numpy as np
from Controllers.set_controller import Controller
from Utils import set_float_precision
from Utils.degree_rad_transformation import rad2degree
from Utils.get_init_slide_mode import InitSlideMode


# 自适应多变量预设性能控制器
class AMPPC(Controller):
    def __init__(self, state_init, command_init, observer):
        super(AMPPC, self).__init__("Adaptive Multi-variable Prescribe Performance Controller", state_init,
                                    command_init,
                                    observer)
        # 控制器参数作为子类变量,方便调试
        self.epsilon = 0.005
        self.epsilon_1 = 0
        self.s0 = InitSlideMode().get_init_mode_state(self.command)
        self.param0 = self.epsilon_1 + np.linalg.norm(self.s0)  # 论文中的epsilon_1 + | | s0 | |
        self.slide_mode_cof = 0.35
        self.theta = 5e-4

    def get_moment(self, time, state, command):
        self.set_state(state)
        self.set_command(command)

        set_float_precision.set_prec(64, 30)
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

        aero_angle_d = self.command[0, 0:3].reshape(-1, 1)
        aero_angle_d_dot = self.command[0, 3:6].reshape(-1, 1)
        aero_angle_2d_dot = self.command[0, 6:9].reshape(-1, 1)

        e1 = np.array([self.state[0, 6], self.state[0, 7], self.state[0, 8]]).reshape(-1, 1) - aero_angle_d
        e2 = R.dot(w) - aero_angle_d_dot

        # print(e1)
        # print(e2)

        # 滑模变量
        s = e1 + np.multiply(self.slide_mode_cof, e2)  # 3×1
        # print("s")
        # print(s)

        s_s = np.tanh(s / self.theta)  # 替代sign函数消除抖动

        R_conj = np.conj(R)
        if not self.observer:
            F = -R @ np.linalg.inv(I) @ Omega @ I @ w - aero_angle_2d_dot + R_conj @ w  # 未添加观测器
        else:
            # print("D")
            D = self.observer.get_D()
            # print(R.shape)
            # print(np.linalg.inv(I).shape)
            # print(D.shape)
            #F = -R @ np.linalg.inv(I) @ Omega @ I @ w - aero_angle_2d_dot + R_conj @ w + D
            F = -R @ np.linalg.inv(I) @ Omega @ I @ w - aero_angle_2d_dot + R_conj @ w
            # print(F.shape)

        u_pie = -e2 - F - np.multiply(np.linalg.norm(s) / (
                self.epsilon + self.param0 * np.exp(-time) - np.linalg.norm(s)), s_s)  # 对应论文中 u'
        u_moment = I @ np.linalg.inv(R) @ u_pie

        # print(-R @ np.linalg.inv(I) @ Omega @ I @ w - aero_angle_2d_dot + R_conj @ w)
        # print(-e2)
        # print(F)
        # print(np.linalg.norm(s) / (
        #         self.epsilon + self.param0 * np.exp(-time) - np.linalg.norm(s)))
        # print(np.multiply(np.linalg.norm(s) / (
        #         self.epsilon + self.param0 * np.exp(-time) - np.linalg.norm(s)), s_s))

        # print("U")
        # print(u_pie.shape)
        # print(u_moment.shape)

        return u_moment

    # 同一被控对象相同
    def get_control_cmd(self, moment):
        # M_x = self.get_moment()[0]
        # M_y = self.get_moment()[1]
        # M_z = self.get_moment()[2]

        # print("get_control_cmd")
        moment = moment.reshape((3, 1))
        M_x = moment[0, 0]
        M_y = moment[1, 0]
        M_z = moment[2, 0]
        # print(M_x)
        # print(M_y)
        # print(M_z)

        # g = 9.80665 / (1 + self.state[0, 0] / (6.356766 * 10 ** 6)) ** 2  # 引力加速度，单位：m / s ^ 2, state[0]为飞行器当前高度
        rho = self.ENVO_CONST.rho0 * np.exp(-self.state[0, 0] * self.ENVO_CONST.hs)  # 大气密度（kg / m ^ 3）
        # print("rho")
        # print(rho)

        # Ma = self.state[0, 3] / self.ATM_CONST.get_atmosphere_params(self.state[0, 0])  # 马赫数
        q_bar = 0.5 * rho * self.state[0, 3] ** 2  # 动压 (pa)

        # print("q_bar")
        # print(q_bar)

        Cl = M_x / (q_bar * self.VH_CONST.S_ref * self.VH_CONST.b_bar)  # 滚转力矩系数
        Cm = M_y / (q_bar * self.VH_CONST.S_ref * self.VH_CONST.c_bar)  # 俯仰力矩系数
        Cn = M_z / (q_bar * self.VH_CONST.S_ref * self.VH_CONST.b_bar)  # 偏航力矩系数

        M_delta = np.array([[self.VH_CONST.Cl_delta_a, 0, self.VH_CONST.Cl_delta_r],
                            [0, self.VH_CONST.Cm_delta_e, 0],
                            [self.VH_CONST.Cn_delta_a, 0, self.VH_CONST.Cn_delta_r]])

        # print(self.VH_CONST.CL_beta * rad2degree(self.state[0, 7]) +
        #       self.VH_CONST.Cl_p * (rad2degree(
        #     self.state[0, 9]) * self.VH_CONST.b_bar) / (2 * self.state[0, 3]) +
        #       self.VH_CONST.Cl_r * (rad2degree(
        #     self.state[0, 11]) * self.VH_CONST.b_bar) / (2 * self.state[0, 3]))
        #
        # print(self.VH_CONST.Cm_0 + self.VH_CONST.Cm_alpha * rad2degree(
        #     self.state[0, 6]) +
        #       self.VH_CONST.Cm_beta * rad2degree(self.state[0, 7]) +
        #       self.VH_CONST.Cm_q * (rad2degree(
        #     self.state[0, 10]) * self.VH_CONST.c_bar / (2 * self.state[0, 3])))
        #
        # print(self.VH_CONST.Cn_beta * rad2degree(self.state[0, 7]) +
        #       self.VH_CONST.Cn_p * (
        #               rad2degree(self.state[0, 9]) * self.VH_CONST.b_bar) / (
        #               2 * self.state[0, 3]) +
        #       self.VH_CONST.Cn_r * (rad2degree(
        #     self.state[0, 11]) * self.VH_CONST.b_bar / (2 * self.state[0, 3])))

        u_degree = np.linalg.inv(M_delta).dot(np.array([[Cl], [Cm], [Cn]]) -
                                              np.array([[self.VH_CONST.CL_beta * rad2degree(self.state[0, 7]) +
                                                         self.VH_CONST.Cl_p * (rad2degree(
                                                  self.state[0, 9]) * self.VH_CONST.b_bar) / (2 * self.state[0, 3]) +
                                                         self.VH_CONST.Cl_r * (rad2degree(
                                                  self.state[0, 11]) * self.VH_CONST.b_bar) / (2 * self.state[0, 3])
                                                         ], [self.VH_CONST.Cm_0 + self.VH_CONST.Cm_alpha * rad2degree(
                                                  self.state[0, 6]) +
                                                             self.VH_CONST.Cm_beta * rad2degree(self.state[0, 7]) +
                                                             self.VH_CONST.Cm_q * (rad2degree(
                                                  self.state[0, 10]) * self.VH_CONST.c_bar / (2 * self.state[0, 3]))
                                                             ], [self.VH_CONST.Cn_beta * rad2degree(self.state[0, 7]) +
                                                                 self.VH_CONST.Cn_p * (
                                                                         rad2degree(self.state[
                                                                                        0, 9]) * self.VH_CONST.b_bar) / (
                                                                         2 * self.state[0, 3]) +
                                                                 self.VH_CONST.Cn_r * (rad2degree(
                                                  self.state[0, 11]) * self.VH_CONST.b_bar / (2 * self.state[0, 3]))
                                                                 ]]
                                                       )
                                              )
        # print("000")
        # print(u_degree)

        dec_lim = 20.0  # 舵偏限幅
        if max(abs(u_degree)) >= dec_lim:
            u_degree = np.multiply(dec_lim, u_degree / max(abs(u_degree)))

        # print("111")
        # print(u_degree.shape)

        return u_degree

# set_float_precision.set_prec(64, 30)
# state_ = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# command_ = np.ones((1, 9))
# M = np.array([[1], [2], [3]])
# observer_ = []
# time_ = 3
# controller = AMPPC(state_, command_, observer_)
# print("moment")
# print(controller.get_moment(time_, state_, command_).shape)
# print(controller.get_control_cmd(M))
#
# # 测试带观测器的控制器
# from Observers.adaptive_slide_mode_observer import AMSDO
# obs_params = np.array((1, 7))
# observer_ = AMSDO(state_, command_, obs_params)
# controller_obs = AMPPC(state_, command_, observer_)
# print(controller_obs.get_moment(time_, state_, command_))
