import numpy as np

from Observers.adaptive_slide_mode_observer import AMSDO


def rk4_d_param(fcn, time, dt, d_D_params, state, command, u_degree, D):
    """
    4th Order Runge-Kutta Integration

    Parameters:
    - fcn: Function to be integrated
    - time: Current time
    - dt: Sample time step
    - d_D_params: Observer parameters diff
    - state: Current state vector
    - command: Commanded state vector
    - U_degree: Control input vector
    - D: Observer state

    Returns:
    - D_params: Updated observer parameters for the next time+dt step
    """

    k1 = fcn(time, d_D_params, state, command, u_degree, D)
    # print("rk4")
    # print(k1.shape)
    # print(d_D_params.shape)
    # print(np.multiply(dt / 2, k1).shape)

    k2 = fcn(time + dt / 2, d_D_params + np.multiply(dt / 2, k1), state, command, u_degree, D)
    k3 = fcn(time + dt / 2, d_D_params + np.multiply(dt / 2, k2), state, command, u_degree, D)
    k4 = fcn(time + dt, d_D_params + np.multiply(dt, k3), state, command, u_degree, D)

    D_params = d_D_params + np.multiply(dt / 6,  (k1 + 2 * k2 + 2 * k3 + k4))

    return D_params

# time_ = 1
# state_init_ = np.ones((1, 12))
# command_init_ = np.ones((1, 9))
# obs_params_init_ = np.ones((1, 7))
# u_degree_ = np.ones((3, 1))
# D_inti = np.ones((3, 1))
# asmdo = AMSDO(state_init_, command_init_, obs_params_init_)
# print(rk4_d_param(asmdo.get_params_diff, time_, 0.001, obs_params_init_, state_init_, command_init_, u_degree_, D_inti).shape)
