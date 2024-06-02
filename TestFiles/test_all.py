from pathlib import Path
import os

from Utils.get_init_slide_mode import InitSlideMode
from VehSimuParams.environment_params import Constants
from VehSimuParams.simulation_params import SimulationParameters, InitialState
from VehSimuParams.vehicle_params import RLVConstants
from Dynamics.attitude_model import RLVAttitudeEquation
from Dynamics.runge4kutta_dynamic import rk4_eom
from TrackingDifferentiator.ADRCTD.td_instance import ADRCTD
from Controllers.adaptive_prescribed_performance_controller import AMPPC
from Observers.adaptive_slide_mode_observer import AMSDO
from Observers.runge4kutta_params import rk4_d_param

import numpy as np
import sys
import time

# 记录高精度的时间开销
start_time = time.perf_counter()

# 项目路径
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
print(parent_dir)

# 创建 VehicleConstants 类实例
vehicle_instance = RLVConstants()
# 创建 Constants 类实例
constants_instance = Constants()
# 创建 SimulationParameters 类实例
simu_params = SimulationParameters()
# 创建 InitialState 类实例
initial_state = InitialState()
# 创建气动模型
RLV_attitude_dynamics = RLVAttitudeEquation()

# 仿真初始化设置
t0 = simu_params.t0  # 仿真零时刻
dt = simu_params.dt  # 积分步长
end = simu_params.tf  # 仿真结束时间
N = 100 # 占位长度，防止溢出

# 获取原始姿态角期望指令 (n,9)
tf_td = np.arange(t0, end + dt, dt)
command = ADRCTD()
raw_command = command.get_raw_command(t0, dt, end)
raw_command_rad = np.multiply(raw_command, np.float64(np.pi / np.float64(180.0)), dtype=np.float64)

# TD预处理指令
td_command = command.get_td_command(t0, dt, end)
td_command_rad = np.multiply(td_command, np.float64(np.pi / 180))
td_command_rad[60000:, 3:6] = 0

# 系统状态变量 (n, 12)，此处为初始值
state_n = np.zeros((len(tf_td) + N, 12))

u_moment_n = np.zeros((3, len(tf_td) + N))  # 控制量, 力矩 (3, n)
u_n = np.zeros((3, len(tf_td)+ N))  # 控制量, 舵偏 (3, n)

D_n = np.zeros((3, len(tf_td)+ N))  # 扰动观测值 (3, n)

obs_params_n = np.zeros((len(tf_td)+ N, 7))  # 观测器中间参数 (n, 7)

# 计步
step_n = 0  # python列表下标从0开始
time_n = step_n * dt

state_n[step_n, :] = np.array([initial_state.alt0, initial_state.phi0, initial_state.theta0, initial_state.V0,
                               initial_state.gamma0, initial_state.chi0, initial_state.aoa0, initial_state.beta0,
                               initial_state.sigma0, initial_state.p0, initial_state.q0, initial_state.r0]).reshape(
    (1, 12))

# 观测器
simu_params.set_OBS('AMSDO')
observer = AMSDO(state_n[step_n, :], td_command_rad[step_n, :], obs_params_n[step_n, :])

# 基础控制器
simu_params.set_controller('AMPPC')
controller = AMPPC(state_n[step_n, :], td_command_rad[step_n, :], observer)

# 滑模初始状态 (线性滑模)
init_slide_mode = InitSlideMode()
s0 = init_slide_mode.get_init_mode_state(td_command_rad[step_n, :].reshape(1, -1))

while time_n <= end:
    # print("STEP")
    print(step_n)
    # u_moment_n[:, step_n]: k1 ; u_n[:, step_n]: k2
    # 计算当前控制量
    k1 = controller.get_moment(time_n, state_n[step_n, :], td_command_rad[step_n, :]).reshape(3)
    u_moment_n[:, step_n] = k1

    k2 = controller.get_control_cmd(k1).reshape(3)
    u_n[:, step_n] = k2

    # 计算下一时刻观测器中间参数及观测扰动值
    k3 = rk4_d_param(observer.get_params_diff, time_n, dt, obs_params_n[step_n, :],
                     state_n[step_n, :], td_command_rad[step_n, :], k2,
                     D_n[:, step_n]).reshape(7)
    obs_params_n[step_n + 1, :] = k3

    k4 = observer.get_disturbances_obs(state_n[step_n, :], k3).reshape(3)
    D_n[:, step_n + 1] = k4

    # 根据控制量与观测值计算下一时刻状态 (六自由度方程龙格库塔积分)
    k5 = rk4_eom(RLV_attitude_dynamics.get_state_diff, time_n, dt, state_n[step_n, :],
                 k2).reshape(12)
    state_n[step_n + 1, :] = k5

    # 更新时序
    step_n = step_n + 1
    time_n = step_n * dt



# 写数据
# f = open(str(parent_dir) + '\\Data\\TDcommand.txt', 'w')
# for i in range(len(td_command_rad[:, 0])):
#     tmp = "{:40.40f}".format(td_command_rad[i, 2])
#     f.write(str(tmp) + ' ')
#     f.write(" " + '\n')
#
# f.close()

# 绘图


# 记录高精度的结束时间
end_time = time.perf_counter()
# 计算并打印代码执行时间
execution_time = end_time - start_time
print(f"Simulation executed in {execution_time} seconds with high precision.")
