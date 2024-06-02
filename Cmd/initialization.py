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

import matplotlib as mpl
from VehSimuParams.simulation_params import SimulationParameters
from matplotlib import pyplot as plt  # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
from pathlib import Path
import os


# 记录高精度的时间开销
start_time = time.perf_counter()

# 获取当前文件的绝对路径及父路径
module_path = os.path.dirname(os.path.realpath(__file__))
script_dir = Path(__file__).parent
parent_dir = script_dir.parent

# 将当前文件所在目录添加到 sys.path 中
sys.path.append(module_path)

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
N = 10  # 防止数组溢出

# 获取原始姿态角期望指令 (n,9)
tf_td = np.arange(t0, end + dt, dt)
command = ADRCTD()
raw_command = command.get_raw_command(t0, dt, end)
raw_command_rad = np.multiply(raw_command, np.float64(pi / 180.0))

# TD预处理指令
td_command = command.get_td_command(t0, dt, end)
td_command[25000:,3:6] = 0;
td_command_rad = np.multiply(td_command, np.float64(pi / 180.0))

# td_command_rad[60000:end, 3] = 0
# td_command_rad[:, 4] = 0
# td_command_rad[60341:end, 5] = 0

# td_command_rad[60000:,3] = 0
# td_command_rad[:,4] = 0
# td_command_rad[60341:, 5] = 0

# First-Order Filter 预处理指令
filter_ = Filter()
filter_command = filter_.get_filter_command(t0, dt, end)
filter_command_rad = np.multiply(filter_command, np.float64(pi / 180.0))

filter_command_rad[60000:,3] = 0
filter_command_rad[:,4] = 0
filter_command_rad[60001:, 5] = 0


# 系统状态变量 (n, 12)，此处为初始值
state_n = np.zeros((len(tf_td) + N, 12))

u_moment_n = np.zeros((3, len(tf_td) + N))  # 控制量, 力矩 (3, n)
u_n = np.zeros((3, len(tf_td) + N))  # 控制量, 舵偏 (3, n)

D_n = np.zeros((3, len(tf_td) + N))  # 扰动观测值 (3, n)

obs_params_n = np.zeros((len(tf_td) + N, 7))  # 观测器中间参数 (n, 7)

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
    #print(step_n)

    # 计算当前控制量
    u_moment_n[:, step_n] = controller.get_moment(time_n, state_n[step_n, :], td_command_rad[step_n, :]).reshape(3)

    u_n[:, step_n] = controller.get_control_cmd(u_moment_n[:, step_n]).reshape(3)

    # 计算下一时刻观测器中间参数及观测扰动值
    obs_params_n[step_n + 1, :] = rk4_d_param(observer.get_params_diff, time_n, dt, obs_params_n[step_n, :],
                                              state_n[step_n, :], td_command_rad[step_n, :], u_n[:, step_n],
                                              D_n[:, step_n]).reshape(7)

    D_n[:, step_n + 1] = observer.get_disturbances_obs(state_n[step_n, :], obs_params_n[step_n + 1, :]).reshape(3)

    # 根据控制量与观测值计算下一时刻状态 (六自由度方程龙格库塔积分)
    state_n[step_n + 1, :] = rk4_eom(RLV_attitude_dynamics.get_state_diff, time_n, dt, state_n[step_n, :],
                                     u_n[:, step_n]).reshape(12)

    # 判断仿真是否结束

    # 更新时序
    step_n = step_n + 1
    time_n = step_n * dt


# AMPPC+ASMO绘图
# 构建绘图所需要的中间变量
mpl.rc("font", family='YouYuan')
mpl.rcParams['text.usetex'] = True  # 默认为false，此处设置为TRUE
plt.rcParams["font.sans-serif"] = ["SimHei"]

#构造误差、滑模面及其包络
aoa                     =  state_n[0:len(tf_td),6]                                       #攻角（rad）
beta                    =  state_n[0:len(tf_td),7]                                       #侧滑角（rad）
sigma                   =  state_n[0:len(tf_td),8]                                       #倾侧角（rad）

p = state_n[0:len(tf_td),9]  # Python 索引从 0 开始
q = state_n[0:len(tf_td),10]
r = state_n[0:len(tf_td),11]

w = np.array([p, q, r])

R = np.zeros((3, 3, len(tf_td)))
for k in range(len(tf_td)):
    aoa_k = aoa[k]
    beta_k = beta[k]
    R[:, :, k] = np.array([
        [-np.cos(aoa_k) * np.tan(beta_k), 1, -np.sin(aoa_k) * np.tan(beta_k)],
        [np.sin(aoa_k), 0, -np.cos(aoa_k)],
        [-np.cos(aoa_k) * np.cos(beta_k), -np.sin(beta_k), -np.sin(aoa_k) * np.cos(beta_k)]
    ])

e1 = np.transpose(state_n[0:len(tf_td),6:9]-td_command_rad[:,0:3])
e2 = np.zeros((3, len(tf_td)))
for k in range(len(tf_td)):
    e2[:, k] = np.dot(R[:, :, k], w[:, k]) - td_command_rad[k, 3:6]
s = e1 + 0.35 * e2

epsilon = 0.005
epsilon_1 = 0
param0 = epsilon_1 + np.abs(s0)

t_n_exp = np.exp(-tf_td)
t_n_times_exp = tf_td * t_n_exp

# 三通道误差各自的包络
e1_envelope_1 = epsilon + (abs(e1[0,:]) - epsilon) * t_n_exp + param0[0] * t_n_times_exp
e1_envelope_2 = epsilon + (abs(e1[1,:]) - epsilon) * t_n_exp + param0[1] * t_n_times_exp
e1_envelope_3 = epsilon + (abs(e1[2,:]) - epsilon) * t_n_exp + param0[2] * t_n_times_exp

e2_envelope_1 = 2*epsilon + (param0[0] + np.abs(e1[0,:]) - epsilon) * t_n_exp + param0[0] * t_n_times_exp
e2_envelope_2 = 2*epsilon + (param0[1] + np.abs(e1[1,:]) - epsilon) * t_n_exp + param0[1] * t_n_times_exp
e2_envelope_3 = 2*epsilon + (param0[2] + np.abs(e1[2,:]) - epsilon) * t_n_exp + param0[2] * t_n_times_exp

s_envelope_1 = epsilon + param0[0] * t_n_exp
s_envelope_2 = epsilon + param0[1] * t_n_exp
s_envelope_3 = epsilon + param0[2] * t_n_exp

# 三通道误差与包络的二范数
e1_norm = np.linalg.norm(e1, axis=0)  # 返回每一列的范数
e2_norm = np.linalg.norm(e2, axis=0)
s_norm = np.linalg.norm(s, axis=0)

param0_norm = epsilon_1 + np.linalg.norm(s0)
e1_norm_col1 = np.linalg.norm(e1[:,0])  # 计算第一列的范数
e1_envelope_norm = epsilon + (e1_norm_col1 - epsilon) * t_n_exp + param0_norm * t_n_times_exp
e2_envelope_norm = 2*epsilon + (param0_norm + e1_norm_col1 - epsilon) * t_n_exp + param0_norm * t_n_times_exp
s_envelope_norm = epsilon + param0_norm * t_n_exp

# 攻角
pparam = dict(xlabel='Time ($s$)', ylabel='Response of Attitude Angle($°$)', title='Angle of Attack')
with plt.style.context(['science', 'ieee', 'no-latex']):
    fig, ax = plt.subplots()
    x = np.arange(0, len(tf_td)) * dt
    # 多条曲线多次调用ax.plot
    ax.plot(x, td_command[:, 0], linestyle='-.', linewidth=1.2, color='#6abc4b', label="Reference")
    ax.plot(x, state_n[0:len(tf_td), 6] * 180.0 / pi, linestyle=':', linewidth=1.2, color='#073762', label="Actual")
    ax.legend(loc="upper right")
    ax.autoscale(tight=True)
    ax.set_xlim(0, 150)
    ax.set_ylim(14, 20)
    ax.set(**pparam)
    ax.grid(ls='--')

    # Note: $\mu$ doesn't work with Times font (used by ieee style)
    # ax.set_ylabel(r'Current ($\mu$A)')
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'Angle of attack command ex5.jpg')
    fig.savefig(relative_path_to_file, dpi=600)

plt.show()

# 侧滑
pparam = dict(xlabel='Time ($s$)', ylabel='Response of Attitude Angle($°$)', title='Sideslip Angle')
with plt.style.context(['science', 'ieee', 'no-latex']):
    fig, ax = plt.subplots()
    x = np.arange(0, len(tf_td)) * dt
    # 多条曲线多次调用ax.plot
    ax.plot(x, td_command[:, 1], linestyle='-.', linewidth=1.2, color='#6abc4b', label="Reference")
    ax.plot(x, state_n[0:len(tf_td), 7] * 180.0 / pi, linestyle=':', linewidth=1.2, color='#073762', label="Actual")
    ax.legend(loc="upper right")
    ax.autoscale(tight=True)
    ax.set_xlim(0, 150)
    ax.set_ylim(-1, 2)
    ax.set(**pparam)
    ax.grid(ls='--')

    # Note: $\mu$ doesn't work with Times font (used by ieee style)
    # ax.set_ylabel(r'Current ($\mu$A)')
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'Sideslip angle command ex5.jpg')
    fig.savefig(relative_path_to_file, dpi=600)

plt.show()

# 倾侧
pparam = dict(xlabel='Time ($s$)', ylabel='Response of Attitude Angle($°$)', title='Bank Angled')
with plt.style.context(['science', 'ieee', 'no-latex']):
    fig, ax = plt.subplots()
    x = np.arange(0, len(tf_td)) * dt
    # 多条曲线多次调用ax.plot  # #073762
    ax.plot(x, td_command[:, 2], linestyle='-', linewidth=1.2, color='#6abc4b', label="Reference")
    ax.plot(x, state_n[0:len(tf_td), 8] * 180.0 / pi, linestyle=':', linewidth=1.2, color='#073762', label="Actual")
    ax.legend(loc="upper right")
    ax.autoscale(tight=True)
    ax.set_xlim(0, 150)
    ax.set_ylim(-2, 12.0)
    ax.set(**pparam)
    ax.grid(ls='--')

    # Note: $\mu$ doesn't work with Times font (used by ieee style)
    # ax.set_ylabel(r'Current ($\mu$A)')
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'Bank angle command ex5.jpg')
    fig.savefig(relative_path_to_file, dpi=600)

plt.show()

# 高度
pparam = dict(xlabel='Time ($s$)', ylabel='Altitude ($m$)', title='Aircraft Altitude')
with plt.style.context(['science', 'ieee', 'no-latex']):
    fig, ax = plt.subplots()
    x = np.arange(0, len(tf_td)) * dt
    ax.plot(x, state_n[0:len(tf_td), 0], linestyle=':', linewidth=1.2, color='#6abc4b')
    ax.legend(loc="upper right")
    ax.autoscale(tight=True)
    ax.set_xlim(0, 150)
    # ax.set_ylim(-2, 12)
    ax.set(**pparam)
    ax.grid(ls='--')

    # Note: $\mu$ doesn't work with Times font (used by ieee style)
    # ax.set_ylabel(r'Current ($\mu$A)')
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'Aircraft altitude ex5.jpg')
    fig.savefig(relative_path_to_file, dpi=600)

plt.show()

# 纬度
pparam = dict(xlabel='Time ($s$)', ylabel='Latitude ($°$)', title='Aircraft Latitude')
with plt.style.context(['science', 'ieee', 'no-latex']):
    fig, ax = plt.subplots()
    x = np.arange(0, len(tf_td)) * dt
    ax.plot(x, state_n[0:len(tf_td), 1] * 180.0 / pi, linestyle=':', linewidth=1.2, color='#6abc4b')
    ax.legend(loc="upper right")
    ax.autoscale(tight=True)
    ax.set_xlim(0, 150)
    # ax.set_ylim(-2, 12)
    ax.set(**pparam)
    ax.grid(ls='--')

    # Note: $\mu$ doesn't work with Times font (used by ieee style)
    # ax.set_ylabel(r'Current ($\mu$A)')
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'Aircraft latitude ex5.jpg')
    fig.savefig(relative_path_to_file, dpi=600)

plt.show()


# 经度
pparam = dict(xlabel='Time ($s$)', ylabel='Longitude ($°$)', title='Aircraft Longitude')
with plt.style.context(['science', 'ieee', 'no-latex']):
    fig, ax = plt.subplots()
    x = np.arange(0, len(tf_td)) * dt
    ax.plot(x, state_n[0:len(tf_td), 2] * 180.0 / pi, linestyle=':', linewidth=1.2, color='#6abc4b')
    ax.legend(loc="upper right")
    ax.autoscale(tight=True)
    ax.set_xlim(0, 150)
    # ax.set_ylim(-2, 12)
    ax.set(**pparam)
    ax.grid(ls='--')

    # Note: $\mu$ doesn't work with Times font (used by ieee style)
    # ax.set_ylabel(r'Current ($\mu$A)')
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'Aircraft longitude ex5.jpg')
    fig.savefig(relative_path_to_file, dpi=600)

plt.show()

# 速度
pparam = dict(xlabel='Time ($s$)', ylabel='Velocity ($m/s$)', title='Aircraft Velocity')
with plt.style.context(['science', 'ieee', 'no-latex']):
    fig, ax = plt.subplots()
    x = np.arange(0, len(tf_td)) * dt
    ax.plot(x, state_n[0:len(tf_td), 3], linestyle=':', linewidth=1.2, color='#6abc4b')
    ax.legend(loc="upper right")
    ax.autoscale(tight=True)
    ax.set_xlim(0, 150)
    ax.set_ylim(4200, 5300)
    ax.set(**pparam)
    ax.grid(ls='--')

    # Note: $\mu$ doesn't work with Times font (used by ieee style)
    # ax.set_ylabel(r'Current ($\mu$A)')
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'Aircraft velocity ex5.jpg')
    fig.savefig(relative_path_to_file, dpi=600)

plt.show()

# 航迹角
pparam = dict(xlabel='Time ($s$)', ylabel='Flight Path Angle ($°$)', title='Flight Path Angle')
with plt.style.context(['science', 'ieee', 'no-latex']):
    fig, ax = plt.subplots()
    x = np.arange(0, len(tf_td)) * dt
    ax.plot(x, state_n[0:len(tf_td), 4], linestyle=':', linewidth=1.2, color='#6abc4b')
    ax.legend(loc="upper right")
    ax.autoscale(tight=True)
    ax.set_xlim(0, 150)
    ax.set_ylim(-0.5, 0.5)
    ax.set(**pparam)
    ax.grid(ls='--')

    # Note: $\mu$ doesn't work with Times font (used by ieee style)
    # ax.set_ylabel(r'Current ($\mu$A)')
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'Flight path angle ex5.jpg')
    fig.savefig(relative_path_to_file, dpi=600)

plt.show()


# 航向角
pparam = dict(xlabel='Time ($s$)', ylabel='Heading Angle ($°$)', title='Heading Angle')
with plt.style.context(['science', 'ieee', 'no-latex']):
    fig, ax = plt.subplots()
    x = np.arange(0, len(tf_td)) * dt
    ax.plot(x, state_n[0:len(tf_td), 5] * 180.0 / pi, linestyle=':', linewidth=1.2, color='#6abc4b')
    ax.legend(loc="upper right")
    ax.autoscale(tight=True)
    ax.set_xlim(0, 150)
    ax.set_ylim(50, 60)
    ax.set(**pparam)
    ax.grid(ls='--')

    # Note: $\mu$ doesn't work with Times font (used by ieee style)
    # ax.set_ylabel(r'Current ($\mu$A)')
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'Heading angle ex5.jpg')
    fig.savefig(relative_path_to_file, dpi=600)

plt.show()

# 副翼舵偏角
pparam = dict(xlabel='Time ($s$)', ylabel='Aileron Declination Angle($°$)', title='Aileron Deflection')
with plt.style.context(['science', 'ieee', 'no-latex']):
    fig, ax = plt.subplots()
    x = np.arange(0, len(tf_td)) * dt
    ax.plot(x, u_n[0, 0:len(tf_td)], linestyle=':', linewidth=1.2, color='#6abc4b')
    ax.legend(loc="upper right")
    ax.autoscale(tight=True)
    ax.set_xlim(0, 150)
    ax.set_ylim(-20, 20)
    ax.set(**pparam)
    ax.grid(ls='--')

    # Note: $\mu$ doesn't work with Times font (used by ieee style)
    # ax.set_ylabel(r'Current ($\mu$A)')
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'Aileron declination ex5.jpg')
    fig.savefig(relative_path_to_file, dpi=600)

plt.show()

# 升降舵偏角
pparam = dict(xlabel='Time ($s$)', ylabel='Elevator Declination Angle($°$)', title='Elevator Deflection')
with plt.style.context(['science', 'ieee', 'no-latex']):
    fig, ax = plt.subplots()
    x = np.arange(0, len(tf_td)) * dt
    ax.plot(x, u_n[1, 0:len(tf_td)], linestyle=':', linewidth=1.2, color='#6abc4b')
    ax.legend(loc="upper right")
    ax.autoscale(tight=True)
    ax.set_xlim(0, 150)
    ax.set_ylim(-20, 15)
    ax.set(**pparam)
    ax.grid(ls='--')

    # Note: $\mu$ doesn't work with Times font (used by ieee style)
    # ax.set_ylabel(r'Current ($\mu$A)')
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'Elevator declination ex5.jpg')
    fig.savefig(relative_path_to_file, dpi=600)

plt.show()

# 方向舵偏角
pparam = dict(xlabel='Time ($s$)', ylabel='Rudder Declination Angle($°$)', title='Rudder Deflection')
with plt.style.context(['science', 'ieee', 'no-latex']):
    fig, ax = plt.subplots()
    x = np.arange(0, len(tf_td)) * dt
    ax.plot(x, u_n[2, 0:len(tf_td)], linestyle=':', linewidth=1.2, color='#6abc4b')
    ax.legend(loc="upper right")
    ax.autoscale(tight=True)
    ax.set_xlim(0, 150)
    ax.set_ylim(-20, 20)
    ax.set(**pparam)
    ax.grid(ls='--')

    # Note: $\mu$ doesn't work with Times font (used by ieee style)
    # ax.set_ylabel(r'Current ($\mu$A)')
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'Rudder declination ex5.jpg')
    fig.savefig(relative_path_to_file, dpi=600)

plt.show()

# 滚转通道扰动观测
pparam = dict(xlabel='Time ($s$)', ylabel='Roll Channel($rad/s^2$)', title='Disturbance Observation Curve')
with plt.style.context(['science', 'ieee', 'no-latex']):
    fig, ax = plt.subplots()
    x = np.arange(0, len(tf_td)) * dt
    ax.plot(x, D_n[0, 0:len(tf_td)], linestyle=':', linewidth=1.2, color='#6abc4b')
    ax.legend(loc="upper right")
    ax.autoscale(tight=True)
    ax.set_xlim(0, 150)
    ax.set_ylim(-2, 2)
    ax.set(**pparam)
    ax.grid(ls='--')

    # Note: $\mu$ doesn't work with Times font (used by ieee style)
    # ax.set_ylabel(r'Current ($\mu$A)')
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'Disturbance observation curve1 ex5.jpg')
    fig.savefig(relative_path_to_file, dpi=600)

plt.show()


# 俯仰通道扰动观测
pparam = dict(xlabel='Time ($s$)', ylabel='Pitch Channel($rad/s^2$)', title='Disturbance Observation Curve')
with plt.style.context(['science', 'ieee', 'no-latex']):
    fig, ax = plt.subplots()
    x = np.arange(0, len(tf_td)) * dt
    ax.plot(x, D_n[1, 0:len(tf_td)], linestyle=':', linewidth=1.2, color='#6abc4b')
    ax.legend(loc="upper right")
    ax.autoscale(tight=True)
    ax.set_xlim(0, 150)
    ax.set_ylim(-1, 1)
    ax.set(**pparam)
    ax.grid(ls='--')

    # Note: $\mu$ doesn't work with Times font (used by ieee style)
    # ax.set_ylabel(r'Current ($\mu$A)')
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'Disturbance observation curve2 ex5.jpg')
    fig.savefig(relative_path_to_file, dpi=600)

plt.show()


# 偏航通道扰动观测
pparam = dict(xlabel='Time ($s$)', ylabel='Yaw Channel($rad/s^2$)', title='Disturbance Observation Curve')
with plt.style.context(['science', 'ieee', 'no-latex']):
    fig, ax = plt.subplots()
    x = np.arange(0, len(tf_td)) * dt
    ax.plot(x, D_n[2, 0:len(tf_td)], linestyle=':', linewidth=1.2, color='#6abc4b')
    ax.legend(loc="upper right")
    ax.autoscale(tight=True)
    ax.set_xlim(0, 150)
    ax.set_ylim(-2, 2)
    ax.set(**pparam)
    ax.grid(ls='--')

    # Note: $\mu$ doesn't work with Times font (used by ieee style)
    # ax.set_ylabel(r'Current ($\mu$A)')
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'Disturbance observation curve3 ex5.jpg')
    fig.savefig(relative_path_to_file, dpi=600)

plt.show()

# 三通道误差 +包络
# 攻角误差+包络
# 侧滑角误差+包络
# 倾侧角误差+包络
fig, axs = plt.subplots(3, 1, figsize=(6, 10),sharex=True,frameon=True)
fig.subplots_adjust(hspace=0.5)
# fig.suptitle('Tracking Error Curves and Envelopes',va='top', ha='center')
pparam1 = dict(xlabel='Time ($s$)', ylabel='Angle of Attack Channel($rad$)', title='Attack Angle Tracking Error')
style = ['science', 'ieee', 'no-latex']
with plt.style.context(style):
    x = np.arange(0, len(tf_td)) * dt
    axs[0].plot(x,e1[0,0:len(tf_td)], linestyle=':', linewidth=1.2, color='#073762',label="Angle of Attack Error")
    axs[0].plot(x, abs(e1_envelope_1), linestyle=':', linewidth=1.2, color='#6abc4b',label="Upper Bound")
    axs[0].plot(x, -abs(e1_envelope_1), linestyle=':', linewidth=1.2, color='#6abc4b',label="Lower Bound")
    axs[0].legend(loc="upper right")
    axs[0].autoscale(tight=True)
    axs[0].set_xlim(0, 150)
    axs[0].set_ylim(-0.05, 0.05)
    axs[0].set(**pparam1)
    axs[0].grid(ls='--')
    axs[0].tick_params(
        axis='both',  # x轴和y轴
        which='both',  # 主刻度和小刻度
        direction='in',  # 刻度向内
    )

# 第二个子图的参数设置
pparam2 = dict(xlabel='Time ($s$)', ylabel='Sideslip Angle Channel($rad$)', title='Sideslip Angle Tracking Error')
with plt.style.context(style):
    x = np.arange(0, len(tf_td)) * dt
    axs[1].plot(x, e1[1, 0:len(tf_td)], linestyle=':', linewidth=1.2, color='#073762', label="Sideslip Angle Error")
    axs[1].plot(x, abs(e1_envelope_2), linestyle=':', linewidth=1.2, color='#6abc4b', label="Upper Bound")
    axs[1].plot(x, -abs(e1_envelope_2), linestyle=':', linewidth=1.2, color='#6abc4b', label="Lower Bound")
    axs[1].legend(loc="upper right")
    axs[1].autoscale(tight=True)
    axs[1].set_xlim(0, 150)
    axs[1].set_ylim(-0.02, 0.02)
    axs[1].set(**pparam2)
    axs[1].grid(ls='--')
    axs[1].tick_params(
        axis='both',  # x轴和y轴
        which='both',  # 主刻度和小刻度
        direction='in',  # 刻度向内
    )

# 第三个子图的参数设置
pparam3 = dict(xlabel='Time ($s$)', ylabel='Bank Angle Channel($rad$)', title='Bank Angle Tracking Error')
with plt.style.context(style):
    x = np.arange(0, len(tf_td)) * dt
    axs[2].plot(x, e1[2, 0:len(tf_td)], linestyle=':', linewidth=1.2, color='#073762', label="Bank Angle Error")
    axs[2].plot(x, abs(e1_envelope_3), linestyle=':', linewidth=1.2, color='#6abc4b', label="Upper Bound")
    axs[2].plot(x, -abs(e1_envelope_3), linestyle=':', linewidth=1.2, color='#6abc4b', label="Lower Bound")
    axs[2].legend(loc="upper right")
    axs[2].autoscale(tight=True)
    axs[2].set_xlim(0, 150)
    axs[2].set_ylim(-0.03, 0.03)
    axs[2].set(**pparam3)
    axs[2].grid(ls='--')
    axs[2].tick_params(
        axis='both',  # x轴和y轴
        which='both',  # 主刻度和小刻度
        direction='in',  # 刻度向内
    )

    # Note: $\mu$ doesn't work with Times font (used by ieee style)
    # ax.set_ylabel(r'Current ($\mu$A)')
relative_path_to_file = os.path.join(parent_dir, 'Data', 'Tracking Error Curves and Envelopes ex5.jpg')
fig.savefig(relative_path_to_file, dpi=600)

plt.show()

# 误差范数 +误差包络范数
pparam = dict(xlabel='Time ($s$)', ylabel='Norm of Tracking Errors($rad$)', title='Norm of Tracking Errors')
with plt.style.context(['science', 'ieee', 'no-latex']):
    fig, ax = plt.subplots()
    x = np.arange(0, len(tf_td)) * dt
    ax.plot(x, e1_norm[0:len(tf_td)], linestyle=':', linewidth=1.2, color='#073762' ,label="Norm of Errors")
    ax.plot(x, e1_envelope_norm[0:len(tf_td)], linestyle=':', linewidth=1.2, color='#6abc4b',label="Perscribed Region")
    ax.legend(loc="upper right")
    ax.autoscale(tight=True)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 0.25)
    ax.set(**pparam)
    ax.grid(ls='--')


    # Note: $\mu$ doesn't work with Times font (used by ieee style)
    # ax.set_ylabel(r'Current ($\mu$A)')
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'Norm of tracking errors ex5.jpg')
    fig.savefig(relative_path_to_file, dpi=600)

plt.show()

# 三通道滑模面 +滑模面包络
# 攻角滑模面+滑模面包络
# 侧滑角滑模面+滑模面包络
# 倾侧角滑模面+滑模面包络
fig, axs = plt.subplots(3, 1, figsize=(6, 10),sharex=True,frameon=True)
fig.subplots_adjust(hspace=0.5)
# fig.suptitle('Sliding Mode Manifold Curves and Envelopes',va='top', ha='center')
pparam1 = dict(xlabel='Time ($s$)', ylabel='Angle of Attack Channel($rad$)', title='Sliding Mode Manifold of Attack Angle Tracking Error')
style = ['science', 'ieee', 'no-latex']
with plt.style.context(style):
    x = np.arange(0, len(tf_td)) * dt
    axs[0].plot(x,s[0,0:len(tf_td)], linestyle=':', linewidth=1.2, color='#073762',label="Sliding Manifold Value")
    axs[0].plot(x, abs(s_envelope_1[0:len(tf_td)]), linestyle=':', linewidth=1.2, color='#6abc4b',label="Upper Bound")
    axs[0].plot(x, -abs(s_envelope_1[0:len(tf_td)]), linestyle=':', linewidth=1.2, color='#6abc4b',label="Lower Bound")
    axs[0].legend(loc="upper right")
    axs[0].autoscale(tight=True)
    axs[0].set_xlim(0, 150)
    axs[0].set_ylim(-0.15, 0.15)
    axs[0].set(**pparam1)
    axs[0].grid(ls='--')
    axs[0].tick_params(
        axis='both',  # x轴和y轴
        which='both',  # 主刻度和小刻度
        direction='in',  # 刻度向内
    )

# 第二个子图的参数设置
pparam2 = dict(xlabel='Time ($s$)', ylabel='Sideslip Angle Channel($rad$)', title='Sliding Mode Manifold of Sideslip Angle Tracking Error')
with plt.style.context(style):
    x = np.arange(0, len(tf_td)) * dt
    axs[1].plot(x,s[1,0:len(tf_td)], linestyle=':', linewidth=1.2, color='#073762',label="Sliding Manifold Value")
    axs[1].plot(x, abs(s_envelope_2[0:len(tf_td)]), linestyle=':', linewidth=1.2, color='#6abc4b',label="Upper Bound")
    axs[1].plot(x, -abs(s_envelope_2[0:len(tf_td)]), linestyle=':', linewidth=1.2, color='#6abc4b',label="Lower Bound")
    axs[1].legend(loc="upper right")
    axs[1].autoscale(tight=True)
    axs[1].set_xlim(0, 150)
    axs[1].set_ylim(-0.04, 0.04)
    axs[1].set(**pparam2)
    axs[1].grid(ls='--')
    axs[1].tick_params(
        axis='both',  # x轴和y轴
        which='both',  # 主刻度和小刻度
        direction='in',  # 刻度向内
    )

# 第三个子图的参数设置
pparam3 = dict(xlabel='Time ($s$)', ylabel='Bank Angle Channel($rad$)', title='Sliding Mode Manifold of Bank Angle Tracking Error')
with plt.style.context(style):
    x = np.arange(0, len(tf_td)) * dt
    axs[2].plot(x,s[2,0:len(tf_td)], linestyle=':', linewidth=1.2, color='#073762',label="Sliding Manifold Value")
    axs[2].plot(x, abs(s_envelope_3[0:len(tf_td)]), linestyle=':', linewidth=1.2, color='#6abc4b',label="Upper Bound")
    axs[2].plot(x, -abs(s_envelope_3[0:len(tf_td)]), linestyle=':', linewidth=1.2, color='#6abc4b',label="Lower Bound")
    axs[2].legend(loc="upper right")
    axs[2].autoscale(tight=True)
    axs[2].set_xlim(0, 150)
    axs[2].set_ylim(-0.2, 0.2)
    axs[2].set(**pparam3)
    axs[2].grid(ls='--')
    axs[2].tick_params(
        axis='both',  # x轴和y轴
        which='both',  # 主刻度和小刻度
        direction='in',  # 刻度向内
    )


    # Note: $\mu$ doesn't work with Times font (used by ieee style)
    # ax.set_ylabel(r'Current ($\mu$A)')
relative_path_to_file = os.path.join(parent_dir, 'Data', 'Sliding Mode Manifold Curves and Envelopes ex5.jpg')
fig.savefig(relative_path_to_file, dpi=600)

plt.show()


# 滑模面范数+滑模面包络范数
pparam = dict(xlabel='Time ($s$)', ylabel='Norm of the Sliding Manifold($rad$)', title='Norm of the Sliding Mode Manifold')
with plt.style.context(['science', 'ieee', 'no-latex']):
    fig, ax = plt.subplots()
    x = np.arange(0, len(tf_td)) * dt
    ax.plot(x, s_norm[0:len(tf_td)], linestyle=':', linewidth=1.2, color='#073762' ,label="Norm of Sliding Manifold Value")
    ax.plot(x, s_envelope_norm[0:len(tf_td)], linestyle=':', linewidth=1.2, color='#6abc4b',label="Perscribed Region")
    ax.legend(loc="upper right")
    ax.autoscale(tight=True)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 0.25)
    ax.set(**pparam)
    ax.grid(ls='--')

    # Note: $\mu$ doesn't work with Times font (used by ieee style)
    # ax.set_ylabel(r'Current ($\mu$A)')
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'Norm of the Sliding Manifold ex5.jpg')
    fig.savefig(relative_path_to_file, dpi=600)

plt.show()

# 记录高精度的结束时间
end_time = time.perf_counter()
# 计算并打印代码执行时间
execution_time = end_time - start_time
print(f"Simulation executed in {execution_time} seconds with high precision.")
