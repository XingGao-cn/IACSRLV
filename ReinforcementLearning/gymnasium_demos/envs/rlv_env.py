# RLV needed
# 需导入类名以访问self变量
from Utils.get_init_slide_mode import InitSlideMode
from VehSimuParams.environment_params import Constants
from VehSimuParams.simulation_params import SimulationParameters, InitialState
from VehSimuParams.vehicle_params import RLVConstants
from Dynamics.attitude_model import RLVAttitudeEquation
from Dynamics.runge4kutta_dynamic import rk4_eom
from TrackingDifferentiator.ADRCTD.td_instance import ADRCTD
from TrackingDifferentiator.Filter.first_order_filter import Filter
from Controllers.adaptive_prescribed_performance_controller import AMPPC
from Observers.adaptive_slide_mode_observer import AMSDO
from Observers.runge4kutta_params import rk4_d_param

import numpy as np
from math import pi
import sys
import time

# DRL needed
import numpy as np
import gymnasium as gym
import torch
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils import seeding
from stable_baselines3.common.env_checker import check_env


# 状态动作统一类型:np.array, np.float65; info类型: dict i.e., {}
class RLVDynamicsEnv(gym.Env):
    metadata = None  # 用于支持可视化的一些设定，改变渲染环境时的参数

    def __init__(self, ep_length: int = 100):
        super(RLVDynamicsEnv, self).__init__()

        # 定义观测空间和动作空间
        # 状态空间根据观测状态进行范围约束
        self.observation_space = spaces.Box(
            low=np.array([-1, -2]).reshape(1, 2),
            high=np.array([2.0, 4.0]).reshape(1, 2),
            shape=(1, 2),
            dtype=np.float64
        )
        # 动作空间严格限制在[-1, 1]之间
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]).reshape(1, 3),
            high=np.array([1.0, 1.0, 1.0]).reshape(1, 3),
            shape=(1, 3),
            dtype=np.float64
        )
        self.n_actions = 3  # 动作个数
        self.state = None  # 当前状态 该状态指飞行状态信息
        self.observation = None  # 若无算法处理, observation ⇔ state
        self.action = None  # 当前action

        # episode settings
        self.ep_length = ep_length  # the length of each episode in timesteps
        self.current_step = 0
        self.num_resets = -1  # Becomes 0 after __init__ exits.

        # 记录仿真时间与计数器
        self.start_clock = None
        self.end_clock = None
        self.step_num = 0

    # Gymnasium在reset中设置seed
    # def seed(self, seed = None):
    #     self.np_random, seed = gym.utils.seeding.np_random(seed)
    #     return [seed]

    # seed固定为常数
    def reset(self, seed=1, options=None):
        # 用于在个episode开始之前重置智能体的状态，把环境恢复到初始状态
        # tuple[ObsType, dict[str, Any]]
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        self.state = self._initial_state()
        return self.state, None  # obs (np.array), info(dict)

    def step(self, action):
        # 应用行动到飞行器动力学模型
        # 接收一个动作，执行这个动作
        # 用来处理状态的转换逻辑, 奖励机制也需要写在这个函数中
        # 返回动作的回报、下一时刻的状态、以及是否结束当前episode及info dict

        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.ep_length

        # info is a dict
        return next_state, immediate_reward, terminated, truncated, info  # ObsType, SupportsFloat, bool, bool, dict[str, Any]

    def _initial_state(self):
        # 返回无人机的初始状态
        # ...
        init_state = np.random.rand(1, 1)
        init_reward = np.random.rand(1, 1)
        init_done = np.random.rand(1, 1)
        init_info = {'info1': 'value1', 'info2': 'value2'}
        return init_state, init_reward, init_done, init_info

    def _get_observation(self):
        pass

    def _apply_action(self, action):
        # 根据行动更新无人机状态并计算奖励
        # tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]
        return

    def _get_reward(self, observation, action):
        return 1.0 if np.all(self.state == action) else 0.0

    def _get_info(self):
        pass


if __name__ == "__main__":
    # To check that the environment follows the Gym interface that SB3 supports
    # Gymnasium also have its own env checker, but it checks a superset of what SB3 supports
    # pytorch,如果安装了tensorflow，则使用from stable_baselines.common.env_checker import check_env
    env = RLVDynamicsEnv()
    check_env(env)  # 无报错则环境接口正常

# if __name__ == '__main__':
#     env = RLVDynamicsEnv()
#     for epoch in range(5):
#         env.reset()
#         print('Epoch', epoch+1, ': ',end='')
#         print(env.state, end='')
#         env.render()    # 刷新画面
#         time.sleep(0.5)
#         for i in range(5):
#             env.step(env.action_space.sample())     # 随机选择一个动作执行
#             print(' -> ', env.state, end='')
#             env.render()    # 刷新画面
#             time.sleep(0.5)
#         print()
#     env.close()
