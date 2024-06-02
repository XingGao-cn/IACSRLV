# 物理参数


class Constants:
    def __init__(self):
        # 圆球模型下的地球平均半径（m）
        self.R0 = 6.378135 * 10 ** 6     # 注：此处和atmosphere_YBW.py中有出入

        # 海平面的平均引力加速度（m/s^2）
        self.g0 = 9.80665

        # 温度（K）
        self.T_SL = 2.8815 * 10 ** 2

        # 压强（Pa）
        self.P_SL = 1.01325 * 10 ** 5

        # 声速（m/s）
        self.a_SL = 3.40294 * 10 ** 2

        # 密度（kg/m^3）
        self.Rho_SL = 1.2250

        # 密度（kg/m^3）
        self.rho0 = 1.2250

        # 大气密度常数（m^(-1)）
        self.hs = 1 / 7110

        # 万有引力常数（m^3/s^2）
        self.GM = self.R0 ** 2 * self.g0

        # 地球自转角速率(rad/s)
        self.omega_e = 7.292 * 10 ** (-5)


# cons  = Constants()
# print(type(cons.R0))