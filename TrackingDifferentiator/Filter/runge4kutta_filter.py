from VehSimuParams.simulation_params import SimulationParameters
import numpy as np

dt = SimulationParameters().dt  # const


def rk4_filter(func, x_d, x_f):
    # func 微分方程， args接收若干参数
    """
       Runge-Kutta 4th order integration for ordinary differential equations.

       Returns:
       - The integrated state at the next time step.
       """

    k1 = func(x_d, x_f)
    k2 = func(x_d + np.multiply(dt / 2, k1), x_f)
    k3 = func(x_d + np.multiply(dt / 2, k2), x_f)
    k4 = func(x_d + np.multiply(dt, k3), x_f)

    x_dot = k1

    x_filter = x_f + np.multiply(dt / 6, (k1 + 2 * k2 + 2 * k3 + k4))

    return x_filter, x_dot  # 返回两个变量




