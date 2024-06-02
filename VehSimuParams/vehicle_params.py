# 飞行器参数, 模型不确定性与外部扰动在此部分进行添加

import numpy as np


# 一类高超声速飞行器参数
# class HVConstants:
#     def __init__(self):
#         # 飞行器参数
#         self.mass = 2200.0  # CAV质量（kg）
#         self.b_bar = 3.6  # 机翼横向参考长度（m）
#         self.c_bar = 3.52  # 机翼平均气动弦长（m）
#         self.S_ref = 12.66 * 0.5  # 参考面积（m^2）
#         self.Ixx = 345.0  # x轴转动惯量（kg*m^2）
#         self.Iyy = 4956.6  # y轴转动惯量（kg*m^2）
#         self.Izz = 4917.1  # z轴转动惯量（kg*m^2）
#         self.Ixz = -130.24  # x轴和z轴耦合惯性积（kg*m^2）
#
#         # 升力系数参数
#         self.CL_0 = -3.39472 * 10 ** -2
#         self.CL_Ma = 6.31965 * 10 ** -5
#         self.CL_alpha = 1.04573 * 10 ** -2
#         self.CL_Ma_alpha = -4.18797 * 10 ** -4
#         self.CL_alpha_2 = 2.17936 * 10 ** -4
#         self.CL_Ma_alpha_2 = 4.14144 * 10 ** -6
#         self.CL_Ma_2_alpha = 9.86629 * 10 ** -6
#         self.CL_alpha_3 = 1.81592 * 10 ** -6
#         self.CL_delta_e = 4.41577 * 10 ** -4
#
#         # 阻力系数参数
#         self.CD_0 = 3.87935 * 10 ** -2
#         self.CD_Ma = -6.07087 * 10 ** -4
#         self.CD_alpha = -9.69344 * 10 ** -4
#         self.CD_Ma_alpha = -5.22735 * 10 ** -5
#         self.CD_alpha_2 = 1.40614 * 10 ** -4
#         self.CD_Ma_alpha_2 = -9.06812 * 10 ** -7
#         self.CD_Ma_2_alpha = 1.66035 * 10 ** -6
#         self.CD_Ma_3 = 7.69399 * 10 ** -7
#         self.CD_alpha_3 = 5.54004 * 10 ** -6
#         self.CD_beta = 1.60289 * 10 ** -4
#
#         # 侧力系数参数
#         self.CY_beta = -9.94962 * 10 ** -3
#         self.CY_Ma_beta = 1.86910 * 10 ** -4
#         self.CY_Ma_2_beta = -4.31664 * 10 ** -6
#         self.CY_delta_r = 2.11233 * 10 ** -4
#         self.CY_Ma_delta_r = -1.58733 * 10 ** -5
#         self.CY_Ma_2_delta_r = 3.92700 * 10 ** -7
#
#         # 滚转力矩系数参数
#         self.Cl_beta = 4.05443 * 10 ** -4
#         self.Cl_Ma_beta = 2.04880 * 10 ** -6
#         self.Cl_delta_r = 3.91986 * 10 ** -3
#         self.Cl_Ma_delta_r = -1.25461 * 10 ** -6
#         self.Cl_delta_a = 6.96636 * 10 ** -3 * 2
#         self.Cl_Ma_delta_a = -2.43573 * 10 ** -6 * 2
#
#         # 俯仰力矩系数参数
#         self.Cm_0 = -1.18572 * 10 ** -2
#         self.Cm_Ma = 1.12428 * 10 ** -5
#         self.Cm_alpha = 2.65247 * 10 ** -3
#         self.Cm_Ma_alpha = -6.94789 * 10 ** -5
#         self.Cm_alpha_2 = -3.76692 * 10 ** -5
#         self.Cm_Ma_alpha_2 = 7.75403 * 10 ** -6
#         self.Cm_Ma_2_alpha = 1.72599 * 10 ** -6
#         self.Cm_alpha_3 = -2.48917 * 10 ** -6
#         self.Cm_Ma_alpha_3 = -1.04605 * 10 ** -7
#         self.Cm_Ma_2_alpha_2 = -1.88603 * 10 ** -7
#         self.Cm_alpha_4 = 1.55695 * 10 ** -7
#         self.Cm_delta_e = 6.69536 * 10 ** -3 * 2
#
#         # 偏航力矩系数参数
#         self.Cn_beta = -2.01578 * 10 ** -3
#         self.Cn_Ma_beta = -5.97591 * 10 ** -5
#         self.Cn_Ma_2_beta = 1.45858 * 10 ** -6
#         self.Cn_delta_r = -1.47621 * 10 ** -3
#         self.Cn_Ma_delta_r = 1.11004 * 10 ** -5
#         self.Cn_Ma_2_delta_r = -2.74658 * 10 ** -7
#
#         # ********************模型不确定性与外部扰动*********************
#         # 三轴力矩系数偏差
#         self.Mx_dis = 0.2  # 滚转力矩扰动百分比
#         self.My_dis = 0.2  # 俯仰力矩扰动百分比
#         self.Mz_dis = 0.2  # 偏航力矩扰动百分比
#
#         # 转动惯量偏差
#         self.Ixx_dis = 0.1  # x轴转动惯量扰动百分比
#         self.Iyy_dis = 0.1  # y轴转动惯量扰动百分比
#         self.Izz_dis = 0.1  # z轴转动惯量扰动百分比
#         self.Ixy_dis = 0.1  # x轴和z轴耦合惯性积扰动百分比
#
#         # 外干扰力矩系数  (具体的形式参照动力学方程或王所论文)
#         self.Dx_1 = 200.0  # 滚转力矩正弦扰动参数1
#         self.Dx_2 = 10  # 滚转力矩正弦扰动参数2
#         self.Dx_3 = 20  # 滚转力矩正弦扰动参数3
#         self.Dy_1 = 500  # 俯仰力矩常值扰动（N*m）
#         self.Dy_2 = 10  # 俯仰力矩正弦扰动参数1
#         self.Dy_3 = 20  # 俯仰力矩正弦扰动参数2
#         self.Dz_1 = 500  # 偏航力矩常值扰动（N*m）
#         self.Dz_2 = 10  # 偏航力矩正弦扰动参数1
#         self.Dz_3 = 20  # 偏航力矩正弦扰动参数2
#
#     # 修改飞行器参数
#     def set_vehicle(self):
#         pass
#
#     # 修改三轴力矩系数偏差
#     def set_torque_coefficient_deviation(self, C):
#         self.Mx_dis = C[0]
#         self.My_dis = C[1]
#         self.Mz_dis = C[2]
#         pass
#
#     # 修改转动惯量偏差
#     def set_inertia_deviation(self, C):
#         self.Ixx_dis = C[0]
#         self.Iyy_dis = C[1]
#         self.Izz_dis = C[2]
#         self.Ixy_dis = C[3]
#         pass
#
#     # 修改外干扰力矩系数
#     def set_external_perturbation_coefficient(self, C):
#         self.Dx_1 = C[0]
#         self.Dx_2 = C[1]
#         self.Dx_3 = C[2]
#         self.Dy_1 = C[3]
#         self.Dy_2 = C[4]
#         self.Dy_3 = C[5]
#         self.Dz_1 = C[6]
#         self.Dz_2 = C[7]
#         self.Dz_3 = C[8]
#         pass


# 一类RLV参数
class RLVConstants:
    def __init__(self):
        # 飞行器参数
        self.mass = 7500.0  # RLV质量（kg）
        self.b_bar = 3.12  # 机翼横向参考长度（m）
        self.c_bar = 1.30  # 机翼平均气动弦长（m）
        self.S_ref = 4.17  # 参考面积（m^2）
        self.Ixx = 885.0  # x轴转动惯量（kg*m^2）
        self.Iyy = 8810.0  # y轴转动惯量（kg*m^2）
        self.Izz = 7770.0  # z轴转动惯量（kg*m^2）
        self.Ixz = -17.40  # x轴和z轴耦合惯性积（kg*m^2）

        # ********************气动力及气动力矩系数*********************
        # 升力系数参数
        self.CL_0 = -4.04009144927135 * 10 ** -1
        self.CL_alpha = 6.99019099340651 * 10 ** -2
        self.CL_beta = 4.79476015299537 * 10 ** -10
        self.CL_delta_e = 2.62888724472374 * 10 ** -3

        # 阻力系数参数
        self.CD_0 = -2.39263528768225 * 10 ** -1
        self.CD_alpha = 3.50894193399550 * 10 ** -2
        self.CD_beta = -8.31115176963430 * 10 ** -10
        self.CD_delta_e = 2.01464783885717 * 10 ** -3

        # 侧力系数参数
        self.CY_beta = 4.628213931313493 * 10 ** -4
        self.CY_delta_r = 3.87542459832595 * 10 ** -3

        # 俯仰力矩系数参数
        self.Cm_0 = -1.51225638003318 * 10 ** -2
        self.Cm_alpha = 1.17127884951365 * 10 ** -2
        self.Cm_beta = 1.1353 * 10 ** -10
        self.Cm_delta_e = -8.3579423913348 * 10 ** -2
        self.Cm_q = 4.168468559969127 * 10 ** -1

        # 滚转力矩系数参数
        self.Cl_beta = 9.8905 * 10 ** -28
        self.Cl_delta_a = 2.2600077191530 * 10 ** -3
        self.Cl_delta_r = 2.8381958692603 * 10 ** -3
        self.Cl_p = 7.9425007625400 * 10 ** -5
        self.Cl_r = 2.20130437741307254 * 10 ** -1

        # 偏航力矩系数参数
        self.Cn_beta = 7.95460076254010 * 10 ** -4
        self.Cn_delta_a = 4.997226656133 * 10 ** -4
        self.Cn_delta_r = 4.1318392189330 * 10 ** -3
        self.Cn_p = 1.419584511117208 * 10 ** -1
        self.Cn_r = 8.562041523648 * 10 ** -4

        # ********************模型不确定性与外部扰动*********************
        # 三轴力矩系数偏差
        self.Mx_dis =  -0.6#-0.5  # 滚转力矩扰动百分比
        self.My_dis = -0.6#-0.5  # 俯仰力矩扰动百分比
        self.Mz_dis = -0.6#-0.5  # 偏航力矩扰动百分比

        # 转动惯量偏差
        self.Ixx_dis = 0.6#0.3  # x轴转动惯量扰动百分比
        self.Iyy_dis = 0.6#0.3  # y轴转动惯量扰动百分比
        self.Izz_dis = 0.6#0.3  # z轴转动惯量扰动百分比
        self.Ixz_dis = 0.6#0.3  # x轴和z轴耦合惯性积扰动百分比

        # 外干扰力矩系数  (具体的形式参照动力学方程或王所论文)
        self.Dx_1 =  200.0  # 滚转力矩正弦扰动（N*m）
        self.Dx_2 =  10  # 滚转力矩正弦扰动参数1
        self.Dx_3 =  20  # 滚转力矩正弦扰动参数2

        self.Dy_1 =  500  # 俯仰力矩常值扰动（N*m）
        self.Dy_2 =  10  # 俯仰力矩正弦扰动参数1
        self.Dy_3 =  20  # 俯仰力矩正弦扰动参数2

        self.Dz_1 =  500  # 偏航力矩常值扰动（N*m）
        self.Dz_2 =  10  # 偏航力矩正弦扰动参数1
        self.Dz_3 =  20  # 偏航力矩正弦扰动参数2

    # 修改飞行器参数
    def set_vehicle(self):
        pass

    # 修改三轴力矩系数偏差
    def set_torque_coefficient_deviation(self, C):
        self.Mx_dis = C[0]
        self.My_dis = C[1]
        self.Mz_dis = C[2]
        pass

    # 修改转动惯量偏差
    def set_inertia_deviation(self, C):
        self.Ixx_dis = C[0]
        self.Iyy_dis = C[1]
        self.Izz_dis = C[2]
        self.Ixy_dis = C[3]
        pass

    # 修改外干扰力矩系数
    def set_external_perturbation_coefficient(self, C):
        self.Dx_1 = C[0]
        self.Dx_2 = C[1]
        self.Dx_3 = C[2]
        self.Dy_1 = C[3]
        self.Dy_2 = C[4]
        self.Dy_3 = C[5]
        self.Dz_1 = C[6]
        self.Dz_2 = C[7]
        self.Dz_3 = C[8]
        pass
