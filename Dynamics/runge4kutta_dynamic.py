# 运动方程4阶Runge-Kutta积分函数
import numpy as np
from Dynamics.attitude_model import RLVAttitudeEquation


def rk4_eom(func, t_n, dt, state_n, u_n):
    # func 一般为动力学方程， args接收若干参数
    """
       Runge-Kutta 4th order integration for ordinary differential equations.

       Parameters:
       - func: The function representing the system's dynamics (ODE).
       - t_n: Current time.
       - dt: Sample step.
       - state_n: Current state.
       - u_n: Control input.

       Returns:
       - The integrated state at the next time step.
       """

    k1 = func(t_n, state_n, u_n)
    k2 = func(t_n + dt / 2, state_n + np.multiply(dt / 2, k1), u_n)
    k3 = func(t_n + dt / 2, state_n + np.multiply(dt / 2, k2), u_n)
    k4 = func(t_n + dt, state_n + np.multiply(dt, k3), u_n)

    integral_state = state_n + np.multiply(dt / 6, (k1 + 2 * k2 + 2 * k3 + k4))

    return integral_state


# tn_ = 30
# dt_ = 0.01
# state_ = np.ones((1, 12))
# u_degree = np.array([[1], [2], [3]])
# RLVeqn = RLVAttitudeEquation()
# print("Runge-Kutta")
# print(rk4_eom(RLVeqn.get_state_diff, tn_, dt_, state_, u_degree).shape)


# References:
# https://blog.csdn.net/haoxun10/article/details/104722414
# https://blog.csdn.net/GODSuner/article/details/117961990
