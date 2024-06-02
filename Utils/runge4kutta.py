# 4阶Runge-Kutta积分函数

def rk4_eom(func, *args, **kwargs):
    # func 一般为动力学方程， args接收若干参数
    """
       Runge-Kutta 4th order integration for ordinary differential equations.

       Parameters:
       - func: The function representing the system's dynamics (ODE).
       - t_n: Current time.
       - dt: Time step.
       - x_n: Current state.
       - u_n: Control input.

       Returns:
       - The integrated state at the next time step.
       """

    k1 = func(args[0], args[2], args[3])
    k2 = func(args[0] + args[1] / 2, args[2] + args[1] / 2 * k1, args[3])
    k3 = func(args[0] + args[1] / 2, args[2] + args[1] / 2 * k2, args[3])
    k4 = func(args[0] + args[1], args[2] + args[1] * k3, args[3])

    integral_value = args[2] + args[1] * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return integral_value


# def rk4_eom(func, t_n, dt, x_n, u_n):
#     k1 = func(t_n, x_n, u_n)
#     k2 = func(t_n + dt/2, x_n + dt/2 * k1, u_n)
#     k3 = func(t_n + dt/2, x_n + dt/2 * k2, u_n)
#     k4 = func(t_n + dt, x_n + dt * k3, u_n)
#
#     X_n_1 = x_n + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
#
#     return X_n_1


# References:
# https://blog.csdn.net/haoxun10/article/details/104722414
# https://blog.csdn.net/GODSuner/article/details/117961990
