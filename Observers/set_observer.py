import numpy as np
from VehSimuParams.vehicle_params import RLVConstants
from VehSimuParams.environment_params import Constants
from Utils import set_float_precision


# 观测器父类
class Observers(object):
    def __init__(self, name, state_init, command_init, obs_params_init):
        set_float_precision.set_prec(64, 30)
        # 基础类变量
        self.name = name  # string类型
        self.state = state_init  # 1×12
        self.obs_params = obs_params_init  # 观测器中间参数
        # 十二状态:
        # [0]高度（m）[1]纬度（rad）[2]经度（rad）[3]速度（m/s）[4]航迹角（rad）[5]航向角（rad）
        # [6]攻角（rad）[7]侧滑角（rad）[8]倾侧角（rad）[9]滚转角速率（rad/s）[10]俯仰角速率（rad/s）[11]偏航角速率（rad/s）
        self.command = command_init  # 观测器设计往往需要用到期望指令

        self.VH_CONST = RLVConstants()  # 飞行器参数常量, 直接赋值
        self.ENVO_CONST = Constants()  # 物理环境参数常量
        self.D = np.zeros((3, 1))  # 观测器属性保存观测值

    # 若设置为私有变量则需要使用以下getter/setter
    def set_D(self, D):
        self.D = np.reshape(D, (3, 1))

    def get_D(self):
        return self.D

    def set_obs_params(self, obs_params):
        self.obs_params = np.reshape(obs_params, (1, -1))

    def get_obs_params(self):
        return self.obs_params

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_state(self, state):
        self.state = np.reshape(state, (1, -1))

    def get_state(self):
        return self.state

    def set_constant(self, constant):
        self.VH_CONST = constant

    def get_constant(self):
        return self.VH_CONST

    def set_command(self, command):
        self.command = np.reshape(command, (1, -1))

    def get_command(self):
        return self.command

    # 根据观测器特性获取扰动观测值
    def get_disturbances_obs(self):
        pass

    # 根据观测器中间参数微分方程
    def get_params_diff(self):
        pass
