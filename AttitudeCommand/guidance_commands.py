# 正确的指令应从TD中导出
# numpy.math和Python标准库的 math.sin的计算速度: https://blog.csdn.net/yeyang911/article/details/12066767
# 在类加载时将指令全部求出, 调用时通过索引返回标量
# 注意！python浮点数不精确，需进行转换np.float64

import numpy as np
from sympy import *
from VehSimuParams import simulation_params


def def_const_command():
    # 常数指令
    comm0 = np.array([20 * pi / 180, 0 * pi / 180, 10 * pi / 180])
    comm1 = np.array([0, 0, 0])
    comm2 = np.array([0, 0, 0])

    comm_ = np.array([comm0, comm1, comm2])

    return comm_


# 定义姿态角指令函数 顺序为：攻角alpha、侧滑角beta、倾侧角gamma
def def_var_command():
    # 关于t的函数指令
    t = symbols('t')
    alpha_f = 0.5 * sin(t)
    beta_f = 0.4 * cos(t)
    gamma_f = 0.3 * sin(t)
    comm0 = np.array([alpha_f, beta_f, gamma_f])

    # 一阶导
    alpha_f_dot = derivation(alpha_f)
    beta_f_dot = derivation(beta_f)
    gamma_f_dot = derivation(gamma_f)
    comm1 = np.array([alpha_f_dot, beta_f_dot, gamma_f_dot])

    # 二阶导
    alpha_f_2dot = derivation(alpha_f_dot)
    beta_f_2dot = derivation(beta_f_dot)
    gamma_f_2dot = derivation(gamma_f_dot)
    comm2 = np.array([alpha_f_2dot, beta_f_2dot, gamma_f_2dot])

    comm = np.array([comm0, comm1, comm2])

    return comm


# 定义函数求导的方法
def derivation(f):
    x = symbols('t')
    d = diff(f, x)

    return d


class RefCommand:
    def __init__(self):
        self.const = simulation_params.SimulationParameters()
        self.reference_command_c = def_const_command()
        self.reference_command_v = self.calculate_command()  # 函数指令
        pass

    # api 返回的值包含期望指令及其一、二阶导
    def get_command(self, t, *args):
        if args[0] == "constant":
            return self.reference_command_c  # 常值指令
            pass

        elif args[0] == "variable":
            return self.reference_command_v[:, :, int(float(format(t / self.const.dt, '.5f')))]
            pass

    def calculate_command(self):
        t = symbols('t')
        comm_v = def_var_command()  # 函数式指令需要进一步计算

        m, n = comm_v.shape
        res_value_v = np.zeros((m, n, int(self.const.tf / self.const.dt) + 1))
        # print(res_value_v, res_value_v.shape)
        for i in range(m):
            for j in range(n):
                res = comm_v[i, j]
                for value in np.arange(self.const.t0, self.const.tf + self.const.dt, self.const.dt):
                    print(i, j, int(float(format(value / self.const.dt, '.5f'))), format(value, '.5f'))
                    res_value_v[i, j, int(float(format(value / self.const.dt, '.5f')))] \
                        = res.evalf(subs={t: format(value, '.5f')})

        return res_value_v


# command = RefCommand()
# value1 = command.get_command(29.7620, "variable")
# print(value1.shape)
