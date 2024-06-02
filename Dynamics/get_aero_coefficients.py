import math

import numpy as np
from VehSimuParams.vehicle_params import RLVConstants
from Utils.degree_rad_transformation import rad2degree


class Aerodynamics(object):
    def __init__(self):
        self.state = np.zeros((1,12))
        self.VH_CONST = RLVConstants()  # 飞行器参数常量,直接赋值
        self.u_a = np.float64(0)  # 滚转等效舵偏角/副翼舵偏角（rad）
        self.u_e = np.float64(0)  # 俯仰等效舵偏角/升降舵偏角（rad）
        self.u_r = np.float64(0)  # 偏航等效舵偏角/方向舵偏角（rad）

    def set_state(self, state):
        self.state = np.reshape(state, (1, -1))

    def get_state(self, state):
        return self.state

    def set_u_degree(self, u):
        self.u_a = u[0]
        self.u_e = u[1]
        self.u_r = u[2]

    def get_u_degree(self):
        return np.array([[self.u_a], [self.u_e], [self.u_r]])

    def updated_aero_coef(self, state, u_degree):
        # 获得最新状态与控制量
        self.set_state(state)
        self.set_u_degree(u_degree)

        # 升力、阻力、侧向力
        CL = np.float64(self.VH_CONST.CL_0 + self.VH_CONST.CL_alpha * rad2degree(
            self.state[0, 6]) + self.VH_CONST.CL_beta * rad2degree(self.state[0, 7]) +
                        self.VH_CONST.CL_delta_e * self.u_e)
        CD = np.float64(self.VH_CONST.CD_0 + self.VH_CONST.CD_alpha * rad2degree(
            self.state[0, 6]) + self.VH_CONST.CD_beta * rad2degree(self.state[0, 7]) +
                        self.VH_CONST.CD_delta_e * self.u_e)
        CY = np.float64(self.VH_CONST.CY_beta * rad2degree(self.state[0, 7]) + self.VH_CONST.CY_delta_r * self.u_r)


        # 滚转、俯仰、偏航
        Cl = np.float64(self.VH_CONST.Cl_beta * rad2degree(
            self.state[0, 7]) + self.VH_CONST.Cl_delta_a * self.u_a + self.VH_CONST.Cl_delta_r * self.u_r +
                        self.VH_CONST.Cl_p * (rad2degree(self.state[0, 9]) * self.VH_CONST.b_bar / (2.0 * self.state[0, 3])) +
                        self.VH_CONST.Cl_r * (rad2degree(self.state[0, 11]) * self.VH_CONST.b_bar / (2.0 * self.state[0, 3]))
                        )
        Cm = np.float64(self.VH_CONST.Cm_0 + self.VH_CONST.Cm_alpha * rad2degree(
            self.state[0, 6]) + self.VH_CONST.Cm_beta * rad2degree(self.state[0, 7]) +
                        self.VH_CONST.Cm_delta_e * self.u_e + self.VH_CONST.Cm_q * (
                                rad2degree(self.state[0, 10]) * self.VH_CONST.c_bar / (2.0 * self.state[0, 3]))
                        )
        Cn = np.float64(self.VH_CONST.Cn_beta * rad2degree(
            self.state[0, 7]) + self.VH_CONST.Cn_delta_a * self.u_a + self.VH_CONST.Cn_delta_r * self.u_r +
                        self.VH_CONST.Cn_p * (rad2degree(self.state[0, 9]) * self.VH_CONST.b_bar / (2.0 * self.state[0, 3])) +
                        self.VH_CONST.Cn_r * (rad2degree(self.state[0, 11]) * self.VH_CONST.b_bar / (2.0 * self.state[0, 3]))
                        )


        return np.array([CL, CD, CY, Cl, Cm, Cn])


# state_ = 2*np.ones((1,12))
# U = np.array([[1], [2], [3]])
# aero = Aerodynamics()
# print(aero.updated_aero_coef(state_, U))
