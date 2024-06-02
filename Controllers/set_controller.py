import numpy as np
import math
from VehSimuParams.vehicle_params import RLVConstants
from VehSimuParams.environment_params import Constants
from Utils import set_float_precision
from Utils.degree_rad_transformation import degree2rad, rad2degree

set_float_precision.set_prec(64, 30)


# 控制器父类
class Controller(object):
    def __init__(self, name, state_init, command_init, observer):
        set_float_precision.set_prec(64, 30)
        # 基础类变量
        self.name = name  # string类型
        self.state = state_init  # 1×12
        # 十二状态:
        # [0]高度（m）[1]纬度（rad）[2]经度（rad）[3]速度（m/s）[4]航迹角（rad）[5]航向角（rad）
        # [6]攻角（rad）[7]侧滑角（rad）[8]倾侧角（rad）[9]滚转角速率（rad/s）[10]俯仰角速率（rad/s）[11]偏航角速率（rad/s）
        self.command = command_init  # 控制器设计往往需要用到期望指令
        self.observer = observer  # 观测器  在此解耦设计，具体代码见Observers

        self.VH_CONST = RLVConstants()  # 飞行器参数常量,直接赋值
        self.ENVO_CONST = Constants()  # 物理环境参数常量
        # self.ATM_CONST = AtmosphereModel()  # 大气模型常量

        self.state = np.reshape(self.state, (1,-1))
        self.command = np.reshape(self.command, (1,-1))

    # 若设置为私有变量则需要使用以下getter/setter
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

    def set_observer(self, observer):
        self.observer = observer

    def get_observer(self):
        return self.observer

    # 根据控制器特性获取控制力矩
    def get_moment(self):
        pass

    # 根据飞行器特性将力矩转换为舵偏角
    def get_control_cmd(self, moment):
        pass

    # 根据观测器获取扰动观测值
    def get_disturbance(self):
        pass


