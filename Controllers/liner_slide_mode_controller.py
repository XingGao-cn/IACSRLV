import math
import numpy as np
from Controllers.set_controller import Controller
from Utils import set_float_precision
from Utils.degree_rad_transformation import rad2degree


# ********************控制器实例*************************
# 线性滑模控制器
class LSM(Controller):
    def __init__(self, state_init, command_init, observer):
        super(LSM, self).__init__("Liner Slide Mode Controller", state_init, command_init, observer)  # 初始化父类以继承属性
        # 控制器参数作为子类变量,方便调试
        self.lambda_s = 5
        self.k_s = 1
        self.lambda_reach = 1
        self.gamma_reach = 0.8

    def get_moment(self, state, command):
        self.set_state(state)
        self.set_command(command)
        # print(self.state.shape)
        set_float_precision.set_prec(64, 30)
        # 状态空间方程系数矩阵
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

        aero_angle_d = self.command[0, 0:3].reshape(3, 1)
        aero_angle_d_dot = self.command[0, 3:6].reshape(3, 1)
        aero_angle_2d_dot = self.command[0, 6:9].reshape(3, 1)

        # print("aero")
        # print(aero_angle_d)
        # print(aero_angle_d_dot)
        # print(aero_angle_2d_dot)

        e_aero_angle = np.array([self.state[0, 6], self.state[0, 7], self.state[0, 8]]).reshape(-1, 1) - aero_angle_d
        e_aero_angle_dot = R.dot(w) - aero_angle_d_dot

        # print("e")
        # print(e_aero_angle)
        # print(e_aero_angle_dot)

        # 滑模变量
        s = e_aero_angle_dot + np.multiply(self.lambda_s, e_aero_angle)  # 3×1
        # print(s.shape)

        # 未添加观测器
        # print("u_mont")
        # print(aero_angle_2d_dot)
        # print(self.lambda_s)
        # print(e_aero_angle_dot)
        # print(np.multiply(self.lambda_s, e_aero_angle_dot))
        #
        # print(aero_angle_2d_dot - np.multiply(self.lambda_s, e_aero_angle_dot))
        # print(-np.multiply(self.k_s, s))
        # print(np.multiply(self.lambda_reach,
        #                                     np.array(
        #                                                    [np.abs(s[0, 0]) ** self.gamma_reach * np.sign(s[0, 0]),
        #                                                     np.abs(s[1, 0]) ** self.gamma_reach * np.sign(s[1, 0]),
        #                                                     np.abs(s[2, 0]) ** self.gamma_reach * np.sign(s[2, 0])
        #                                                     ])))

        u_moment = I @ np.linalg.inv(R) @ (R @ np.linalg.inv(I) @ Omega @ I @ w +
                                           aero_angle_2d_dot - np.multiply(self.lambda_s, e_aero_angle_dot) -
                                           np.multiply(self.k_s, s) -
                                           np.multiply(self.lambda_reach,
                                                       np.array(
                                                           [[np.abs(s[0, 0]) ** self.gamma_reach * np.sign(s[0, 0])],
                                                            [np.abs(s[1, 0]) ** self.gamma_reach * np.sign(s[1, 0])],
                                                            [np.abs(s[2, 0]) ** self.gamma_reach * np.sign(s[2, 0])]
                                                            ]))
                                           )

        return u_moment

    def get_control_cmd(self, M):
        # M_x = self.get_moment()[0]
        # M_y = self.get_moment()[1]
        # M_z = self.get_moment()[2]
        M_x = M[0, 0]
        M_y = M[1, 0]
        M_z = M[2, 0]

        # g = 9.80665 / (1 + self.state[0, 0] / (6.356766 * 10 ** 6)) ** 2  # 引力加速度，单位：m / s ^ 2, state[0]为飞行器当前高度
        rho = self.ENVO_CONST.rho0 * np.exp(-self.state[0, 0] * self.ENVO_CONST.hs)  # 大气密度（kg / m ^ 3）
        # Ma = self.state[0, 3] / self.ATM_CONST.get_atmosphere_params(self.state[0, 0])  # 马赫数
        q_bar = 0.5 * rho * self.state[0, 3] ** 2  # 动压 (pa)

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

        dec_lim = 20  # 舵偏限幅
        if max(abs(u_degree)) >= dec_lim:
            u_degree = u_degree / max(abs(u_degree)) * dec_lim

        return u_degree


# set_float_precision.set_prec(64, 30)
# state_ = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# command_ = np.ones((1, 9))
# M = np.array([[1], [2], [3]])
# observer_ = []
# controller = LSM(state_, command_, observer_)
# print(controller.get_moment(state_, command_))
# print(controller.get_control_cmd(M))