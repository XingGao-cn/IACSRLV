# 测试跟踪微分器

import decimal
import numpy as np
from TrackingDifferentiator.ADRCTD import command_2
from TrackingDifferentiator.ADRCTD import td

# 绘图所用packets
import numpy as np
import matplotlib as mpl
from scipy import interpolate
from VehSimuParams.simulation_params import SimulationParameters
from matplotlib import pyplot as plt # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
from pathlib import Path
import os

script_dir = Path(__file__).parent
parent_dir = script_dir.parent.parent


class ADRCTD():
    def __init__(self):
        # 基础类变量 如无特殊设置, 参数遵循SimulationParameters
        self.t0 = SimulationParameters().t0
        self.dt =  SimulationParameters().dt
        self.end = SimulationParameters().tf
        self.rad_command = np.zeros((1, 1))
        self.td_command = np.zeros((1, 1))

    def get_raw_command(self, t0, dt, end):
        self.t0 = t0
        self.dt = dt
        self.end = end

        tf_td = np.arange(self.t0, self.end + self.dt, self.dt)
        v = np.zeros((len(tf_td), 3))
        # 获取原始指令
        for i in range(len(tf_td)):
            tmp = command_2.get_command2(tf_td[i], end)
            v[i, :] = tmp[:3]
        self.rad_command = v

        return self.rad_command

    def get_td_command(self, t0, dt, end):
        v = self.get_raw_command(t0, dt, end)
        # 通道一(攻角)通过 TD 后的指令以及一阶，二阶导数
        v1 = np.zeros_like(v[:, 0])
        v1_dot = np.zeros_like(v[:, 0])
        v1_d_dot = np.zeros_like(v[:, 0])

        td_tmp1 = td.TDIterator(v[0, 0])
        for i in range(len(v)):
            v1[i], v1_dot[i] = td_tmp1.TD(v[i, 0], dt)

        td_tmp2 = td.TDIterator(v1_dot[0])
        for i in range(len(v)):
            _, v1_d_dot[i] = td_tmp2.TD(v1_dot[i], dt)

        # 通道二(侧滑)通过TD后的指令以及一阶，二阶导数
        v2 = np.zeros_like(v[:, 1])
        v2_dot = np.zeros_like(v[:, 1])
        v2_d_dot = np.zeros_like(v[:, 1])

        td_tmp3 = td.TDIterator(v[0, 1])
        for i in range(len(v)):
            v2[i], v2_dot[i] = td_tmp3.TD(v[i, 1], dt)

        td_tmp4 = td.TDIterator(v2_dot[0])
        for i in range(len(v)):
            _, v2_d_dot[i] = td_tmp4.TD(v2_dot[i], dt)

        # 通道三(倾侧)通过TD后的指令以及一阶，二阶导数
        v3 = np.zeros_like(v[:, 2])
        v3_dot = np.zeros_like(v[:, 2])
        v3_d_dot = np.zeros_like(v[:, 2])

        td_tmp5 = td.TDIterator(v[0, 2])
        for i in range(len(v)):
            v3[i], v3_dot[i] = td_tmp5.TD(v[i, 2], dt)

        td_tmp6 = td.TDIterator(v3_dot[0])
        for i in range(len(v)):
            _, v3_d_dot[i] = td_tmp6.TD(v3_dot[i], dt)

        # TD_command = np.zeros((len(v[:, 0]), 9))
        self.td_command = np.column_stack((v1, v2, v3, v1_dot, v2_dot, v3_dot, v1_d_dot, v2_d_dot, v3_d_dot))

        return self.td_command




# TD预处理指令
# t0 = 0
# dt = 0.001  # 积分步长
# end = 150  # 仿真结束时间
# tf_td = np.arange(t0, end + dt, dt)
#
# command = ADRCTD()
# Raw_command = command.get_raw_command(t0, dt, end)
# TD_command = command.get_td_command(t0, dt, end)
#
# Raw_command = np.multiply(Raw_command,np.float64(np.pi/180))
# TD_command = np.multiply(TD_command,np.float64(np.pi/180))
# tf_td = np.arange(t0, end + dt, dt)
# v = np.zeros((len(tf_td), 3))
#
# # 获取原始指令
# for i in range(len(tf_td)):
#     tmp = command_2.get_command2(tf_td[i], end)
#     v[i, :] = tmp[:3]
#
# # 通道一(攻角)通过 TD 后的指令以及一阶，二阶导数
# v1 = np.zeros_like(v[:, 0])
# v1_dot = np.zeros_like(v[:, 0])
# v1_d_dot = np.zeros_like(v[:, 0])
#
# td_tmp1 = td.TDIterator(v[0, 0])
# for i in range(len(v)):
#     v1[i], v1_dot[i] = td_tmp1.TD(v[i, 0], dt)
#
# td_tmp2 = td.TDIterator(v1_dot[0])
# for i in range(len(v)):
#     _, v1_d_dot[i] = td_tmp2.TD(v1_dot[i], dt)
#
# # 通道二(侧滑)通过TD后的指令以及一阶，二阶导数
# v2 = np.zeros_like(v[:, 1])
# v2_dot = np.zeros_like(v[:, 1])
# v2_d_dot = np.zeros_like(v[:, 1])
#
# td_tmp3 = td.TDIterator(v[0, 1])
# for i in range(len(v)):
#     v2[i], v2_dot[i] = td_tmp3.TD(v[i, 1], dt)
#
# td_tmp4 = td.TDIterator(v2_dot[0])
# for i in range(len(v)):
#     _, v2_d_dot[i] = td_tmp4.TD(v2_dot[i], dt)
#
# # 通道三(倾侧)通过TD后的指令以及一阶，二阶导数
# v3 = np.zeros_like(v[:, 2])
# v3_dot = np.zeros_like(v[:, 2])
# v3_d_dot = np.zeros_like(v[:, 2])
#
# td_tmp5 = td.TDIterator(v[0, 2])
# for i in range(len(v)):
#     v3[i], v3_dot[i] = td_tmp5.TD(v[i, 2], dt)
#
# td_tmp6 = td.TDIterator(v3_dot[0])
# for i in range(len(v)):
#     _, v3_d_dot[i] = td_tmp6.TD(v3_dot[i], dt)
#
# TD_command = np.zeros((len(v[:, 0]), 9))
# TD_command = np.column_stack((v1, v2, v3, v1_dot, v2_dot, v3_dot, v1_d_dot, v2_d_dot, v3_d_dot))
# TD_command = np.multiply(TD_command,np.float64(np.pi/180))


# 指令绘图及保存

# parent_dir = os.path.join(parent_dir, 'Data')
# x = np.linspace(t0, end + dt, np.int64(end / dt))
# f = open(parent_dir+'\\command.txt', 'w')
# for i in range(len(TD_command[:,0])):
#     tmp = "{:40.40f}".format(TD_command[i, 0])
#     f.write(str(tmp) + '\n')
#
# f.close()

# mpl.rc("font", family='YouYuan')
# mpl.rcParams['text.usetex'] = True  # 默认为false，此处设置为TRUE
# plt.rcParams["font.sans-serif"] = ["SimHei"]
#
# pparam = dict(xlabel='Time (s)', ylabel='Altitude tracking command ($°$)', title='Angle of attack command')
# with plt.style.context(['science', 'ieee', 'no-latex']):
#     fig, ax = plt.subplots()
#     x = np.arange(0, len(tf_td))*0.001
#     ax.plot(x, TD_command[:, 0], linestyle=':', linewidth =1.5, color='darkgoldenrod',label="Desired command")
#     ax.legend(loc="upper right")
#     ax.autoscale(tight=True)
#     ax.set_xlim(0, 150)
#     ax.set_ylim(12, 20)
#     ax.set(**pparam)
#     ax.grid(ls='--')
#
#
#     # Note: $\mu$ doesn't work with Times font (used by ieee style)
#     # ax.set_ylabel(r'Current ($\mu$A)')
#     relative_path_to_file = os.path.join(parent_dir, 'Data', 'Angle of attack command.jpg')
#     fig.savefig(relative_path_to_file, dpi=600)
#
# plt.show()
#
# pparam = dict(xlabel='Time (s)', ylabel='Altitude tracking command ($°$)', title='Sideslip angle command')
# with plt.style.context(['science', 'ieee', 'no-latex']):
#     fig, ax = plt.subplots()
#     x = np.arange(0, len(tf_td))*0.001
#     ax.plot(x, TD_command[:, 1], linestyle=':', linewidth =1.5, color='#fc0611',label="Desired command")
#     ax.legend(loc="upper right")
#     ax.autoscale(tight=True)
#     ax.set_xlim(0, 150)
#     ax.set_ylim(-5, 5)
#     ax.set(**pparam)
#     ax.grid(ls='--')
#
#
#     # Note: $\mu$ doesn't work with Times font (used by ieee style)
#     # ax.set_ylabel(r'Current ($\mu$A)')
#     relative_path_to_file = os.path.join(parent_dir, 'Data', 'Sideslip angle command.jpg')
#     fig.savefig(relative_path_to_file, dpi=600)
#
# plt.show()
#
# pparam = dict(xlabel='Time (s)', ylabel='Altitude tracking command ($°$)', title='Roll angle command')
# with plt.style.context(['science', 'ieee', 'no-latex']):
#     fig, ax = plt.subplots()
#     x = np.arange(0, len(tf_td))*0.001
#     ax.plot(x, TD_command[:, 2], linestyle=':', linewidth =1.5, color='#6abc4b',label="Desired command")
#     ax.legend(loc="upper right")
#     ax.autoscale(tight=True)
#     ax.set_xlim(0, 150)
#     ax.set_ylim(-2, 12)
#     ax.set(**pparam)
#     ax.grid(ls='--')
#
#
#     # Note: $\mu$ doesn't work with Times font (used by ieee style)
#     # ax.set_ylabel(r'Current ($\mu$A)')
#     relative_path_to_file = os.path.join(parent_dir, 'Data', 'Roll angle command.jpg')
#     fig.savefig(relative_path_to_file, dpi=600)
#
# plt.show()

# 参考文献
# [1]武利强,林浩,韩京清.跟踪微分器滤波性能研究[J].系统仿真学报, 2004, 16(4):3.DOI:10.3969/j.issn.1004-731X.2004.04.012.
# [2]https://blog.csdn.net/m0_37764065/article/details/108668033.  %复制链接得去掉最后的.
