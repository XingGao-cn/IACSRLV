# 杨炳尉气模型
#      公式计算值与原表的相对误差小于万分之三。用于弹道分析、再入分析、制导、气动力等的计算,能节省机时和内存。更便于在袖珍计算机上使用
#      考虑重力场随高度变化造成的影响，引入重力位势高度 H，替几何高度 Z，
#      适用于-5-91km，若超出此范围则取最近的边界对应的值
#      输入几何高度（单位：m）H，输出为对应的温度 T（单位：K）、压强 P（单位：Pa）、密度 Rho（单位：kg/m^3）、声速 S（单位：m/s^2）、引力加速度 G（圆球模型下，单位：m/s^2）
# 🔺此文件与 alt、Z相关的量均可转为非矩阵形式的变量

import numpy as np
from VehSimuParams import environment_params


class AtmosphereModel:
    def __init__(self):
        # 海平面大气参数值
        self.constants = environment_params.Constants()
        self.T_SL = self.constants.T_SL  # 温度，单位：K
        self.P_SL = self.constants.P_SL  # 压强，单位：Pa
        self.Rho_SL = self.constants.Rho_SL  # 密度，单位：kg/m^3
        self.g_SL = self.constants.g0  # 引力加速度，单位：m/s^2
        self.a_SL = self.constants.a_SL  # 声速，单位：m/s
        self.R0 = 6.356766 * 10 ** 3  # 地球半径，单位：km   #注：此处和environment_param中参数有出入 非圆球模型
        # self.R0 = self.constants.R0

    def get_atmosphere_params(self, alt):
        Z = alt / 1000  # 几何高度，单位：km
        H = Z / (1 + Z / self.R0)  # 重力位势高度H与几何高度Z的转换

        G = 9.80665 / (1 + Z / self.R0) ** 2  # 引力加速度，单位：m/s^2

        W = np.zeros((1,1))

        # 变量初始化
        T, P, Rho, S = np.zeros_like(Z), np.zeros_like(Z), np.zeros_like(Z), np.zeros_like(Z)

        # 检查高度范围
        if Z < -5 or Z > 91:
            print("🔺警告！飞行范围超出模型范围..")
            pass  # 超出适用范围，为防止仿真中断，取最近可行值

            # 裁剪高度
            Z = np.clip(Z, -5, 91)
            H = Z / (1 + Z / self.R0)  # 重力位势高度H与几何高度Z的转换

            # 根据几何高度划分
        if -5 <= Z <= 11.0191:
            W = 1 - H / 44.3308
            T = 288.15 * W  # 温度，单位：K
            P = W ** 5.2559 * self.P_SL  # 压强，单位：Pa
            Rho = W ** 4.2559 * self.Rho_SL  # 密度，单位：kg/m^3
        elif 11.0191 < Z <= 20.0631:
            W = np.exp((14.9647 - H) / 6.3416)
            T = 216.650  # 温度，单位：K
            P = 1.1953 * 10 ** (-1) * W * self.P_SL  # 压强，单位：Pa
            Rho = 1.5898 * 10 ** (-1) * W * self.Rho_SL  # 密度，单位：kg/m^3
        # ... waiting
        elif 20.0631 < Z <= 32.1619:
            W = 1 + (H - 24.9021) / 221.552
            T = 221.552 * W  # 温度，单位：K
            P = 2.5158e-3 * W ** (-34.1629) * self.P_SL  # 压强，单位：Pa
            Rho = 3.2722e-2 * W ** (-35.1629) * self.Rho_SL  # 密度，单位：kg/m^3
        elif 32.1619 < Z <= 47.3501:
            W = 1 + (H - 39.7499) / 89.4107
            T = 250.350 * W  # 温度，单位：K
            P = 2.8338e-3 * W ** (-12.2011) * self.P_SL  # 压强，单位：Pa
            Rho = 3.2618e-3 * W ** (-13.2011) * self.Rho_SL  # 密度，单位：kg/m^3
        elif 47.3501 < Z <= 51.4125:
            W = np.exp((48.6252 - H) / 7.9223)
            T = 270.650  # 温度，单位：K
            P = 8.9155e-4 * W * self.P_SL  # 压强，单位：Pa
            Rho = 9.4920e-4 * W * self.Rho_SL  # 密度，单位：kg/m^3
        elif 51.4125 < Z <= 71.8020:
            W = 1 - (H - 59.4390) / 88.2218
            T = 247.021 * W  # 温度，单位：K
            P = 2.1671e-4 * W ** 12.2011 * self.P_SL  # 压强，单位：Pa
            Rho = 2.5280e-4 * W ** 11.2011 * self.Rho_SL  # 密度，单位：kg/m^3
        elif 71.8020 < Z < 86.0000:
            W = 1 - (H - 78.0303) / 100.2950
            T = 200.590 * W  # 温度，单位：K
            P = 1.2274e-5 * W ** 17.0816 * self.P_SL  # 压强，单位：Pa
            Rho = 1.7632e-5 * W ** 16.0816 * self.Rho_SL  # 密度，单位：kg/m^3
        elif 86.0000 <= Z <= 91.0000:
            W = np.exp((87.2848 - H) / 5.4700)
            T = 186.870  # 温度，单位：K
            P = (2.2730 + 1.042e-3 * H) * 1e-6 * W * self.P_SL  # 压强，单位：Pa
            Rho = 3.6411e-6 * W * self.Rho_SL  # 密度，单位：kg/m^3

        # 声速与温度有关
        S = 20.0468 * np.sqrt(T)  # 声速，单位：m/s

        return T, P, Rho, S, G


# atm = AtmosphereModel()
# print(atm.get_atmosphere_params(50000))


# 参考文献
# 杨炳尉.标准大气参数的公式表示[J].宇航学报, 1983(01):86-89.DOI:CNKI:SUN:YHXB.0.1983-01-009.
