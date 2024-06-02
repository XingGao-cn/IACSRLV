import numpy as np
from VehSimuParams.simulation_params import InitialState
import math


class InitSlideMode():
    def __init__(self):
        self.init_state = InitialState()
        # 加载通过跟踪微分器后的初始时刻指令
        self.command_init = np.ones((1, 9))

    def set_command(self, command):
        self.command_init = command
        self.command_init.reshape(1, -1)

    def get_init_mode_state(self, command):
        self.set_command(command)
        p_init = self.init_state.p0  # 初始滚转角速率（rad/s）
        q_init = self.init_state.q0  # 初始俯仰角速率（rad/s）
        r_init = self.init_state.r0  # 初始偏航角速率（rad/s）

        R = np.array([[-math.cos(self.init_state.aoa0) * math.tan(self.init_state.beta0), 1,
                       -math.sin(self.init_state.aoa0) * math.tan(self.init_state.beta0)],
                      [math.sin(self.init_state.aoa0), 0, -math.cos(self.init_state.beta0)],
                      [-math.cos(self.init_state.aoa0) * math.cos(self.init_state.beta0),
                       -math.sin(self.init_state.beta0),
                       -math.sin(self.init_state.aoa0) * math.cos(self.init_state.beta0)]])

        w = np.array([[p_init], [q_init], [r_init]])

        # ei_0
        e1_init = np.array(
            [[self.init_state.aoa0], [self.init_state.beta0], [self.init_state.sigma0]]) - self.command_init[0,
                                                                                           0:3].reshape(-1, 1)
        e2_inti = R.dot(w) - self.command_init[0, 3:6].reshape(-1, 1)  # R*w -omega_d_dot

        # print("e0")
        # print(e1_init)
        # print(e2_inti)

        # 线性滑模
        s_init = e1_init + e2_inti

        print("s0")
        print(s_init)

        return s_init


# init_slide_mode = InitSlideMode()
# init_slide_mode.get_init_mode_state(np.ones((1, 9)))
