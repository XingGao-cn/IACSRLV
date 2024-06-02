import numpy as np
from sympy import *


# # 定义函数求导的方法
# def derivation(y):
#     x = symbols('t')
#     d = diff(y, x)
#     return d
#
#
# x = symbols('x')
# y = (x - 9) ** 2
#
# # 求导
# res = derivation(y)
# print("求导后的函数结果为:", res)
#
# # 代入具体值
# value = 13
# res_value = res.evalf(subs={x: value})
# print("向求导后的函数中代入值:", res_value)
#
# # 关于t的函数指令
# t = symbols('t')
# alpha_f = 0.5 * sin(t)
# beta_f = 0.4 * cos(t)
# gamma_f = 0.3 * sin(t)
# comm0 = np.array([alpha_f, beta_f, gamma_f])
#
# # 一阶导
# alpha_f_dot = derivation(alpha_f)
# beta_f_dot = derivation(beta_f)
# gamma_f_dot = derivation(gamma_f)
# comm1 = np.array([alpha_f_dot, beta_f_dot, gamma_f_dot])
#
# comm2 = np.array([comm0, comm1])
# print(comm2.shape)
#
# m, n = comm2.shape
# res_value = np.zeros((m, n, 20))
#
# print(res_value, res_value.shape)
# for i in range(m):
#     for j in range(n):
#         print(i, j, "\n")
#         res = comm2[i, j]
#         print(res)
#         for value in np.arange(0, 2 + 0.1, 0.1):
#             print(i, j, int(value*10))
#             res_value[i, j, int(value)] = res.evalf(subs={t: value})
# print(res_value.shape, res_value[:, :, 19])


import numpy as np
import matplotlib as mpl
from scipy import interpolate
from matplotlib import pyplot as plt
from pathlib import Path
import os

script_dir = Path(__file__).parent
parent_dir = script_dir.parent


mpl.rc("font",family='YouYuan')
from matplotlib.font_manager import FontManager
import subprocess

# https://zhuanlan.zhihu.com/p/148233372
# https://zhuanlan.zhihu.com/p/104081310
plt.style.use('ieee')
# plt.style.use(['science','ieee'])

plt.rc('font', family='times new roman')
mpl.rc('font', family='times new roman')
# plt.style.use('fivethirtyeight')
mpl.rcParams['text.usetex'] = True#默认为false，此处设置为TRUE
plt.rcParams["font.sans-serif"] = ["SimHei"]
x = [0, 1, 5, 10, 15, 20, 30, 40, 50, 60, 65, 70]
y1 = [-3.26, -3.07, -2.28, -1.27, -0.33, 0.47, 1.64, 2.36, 2.67, 2.73, 2.72, 2.68]
y2 = [-3.256, -3.064, -2.283, -1.273, -0.332, 0.468, 1.638, 2.363, 2.675, 2.735, 2.716, 2.681]

# 样条插值
tck, u = interpolate.splprep([x, y1], s=0)
unew = np.arange(0, 1.01, 0.01)
out = interpolate.splev(unew, tck)

tck, u = interpolate.splprep([x, y2], s=0)
out1 = interpolate.splev(unew, tck)

# 绘制
with plt.style.context(['high-vis', 'no-latex']):
    fig, ax = plt.subplots()
    ax.plot(out[0], out[1], label='smartload')
    ax.plot(out1[0], out1[1], label='lr')
    ax.legend(title=r"Order")
    ax.set(xlabel='Heel angle (°)')
    ax.set(ylabel='Gz (m) ')
    ax.autoscale(tight=True)  # 设置Y刻度最大值
    ax.set_ylim(top=3)
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'fig1.jpg')
    fig.savefig(relative_path_to_file, dpi=600)

plt.show()


def model(x, p):
    return x ** (2 * p + 1) / (1 + x ** (2 * p))


pparam = dict(xlabel='Voltage (mV)', ylabel='Current ($\mu$A)')

x = np.linspace(0.75, 1.25, 201)

with plt.style.context(['science']):
    fig, ax = plt.subplots()
    for p in [10, 15, 20, 30, 50, 100]:
        ax.plot(x, model(x, p), label=p)
    ax.legend(title='Order')
    ax.autoscale(tight=True)
    ax.set(**pparam)
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'fig2.jpg')
    fig.savefig(relative_path_to_file, dpi=600)

plt.show()

with plt.style.context(['science', 'ieee', 'no-latex']):
    fig, ax = plt.subplots()
    for p in [10, 20, 40, 100]:
        ax.plot(x, model(x, p), label=p)
    ax.legend(title='Order')
    ax.autoscale(tight=True)
    ax.set(**pparam)
    # Note: $\mu$ doesn't work with Times font (used by ieee style)
    ax.set_ylabel(r'Current ($\mu$A)')
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'fig2a.jpg')
    fig.savefig(relative_path_to_file, dpi=600)


with plt.style.context(['science', 'ieee', 'std-colors','no-latex']):
    fig, ax = plt.subplots()
    for p in [10, 15, 20, 30, 50, 100]:
        ax.plot(x, model(x, p), label=p)
    ax.legend(title='Order')
    ax.autoscale(tight=True)
    ax.set(**pparam)
    # Note: $\mu$ doesn't work with Times font (used by ieee style)
    ax.set_ylabel(r'Current  ($\mu$A)')
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'fig2b.jpg')
    fig.savefig(relative_path_to_file, dpi=400)

plt.show()

with plt.style.context(['science', 'nature','no-latex']):
    fig, ax = plt.subplots()
    for p in [10, 15, 20, 30, 50, 100]:
        ax.plot(x, model(x, p), label=p)
    ax.legend(title='Order')
    ax.autoscale(tight=True)
    ax.set(**pparam)
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'fig2c.jpg')
    fig.savefig(relative_path_to_file, dpi=600)

plt.show()

# with latex
with plt.style.context(['science', 'scatter']):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([-2, 2], [-2, 2], 'k--')
    ax.fill_between([-2, 2], [-2.2, 1.8], [-1.8, 2.2],
                    color='dodgerblue', alpha=0.2, lw=0)
    for i in range(7):
        x1 = np.random.normal(0, 0.5, 10)
        y1 = x1 + np.random.normal(0, 0.2, 10)
        ax.plot(x1, y1, label=r"$^\#${}".format(i + 1))
    lgd = r"$\mathring{P}=\begin{cases}1&\text{if $\nu\geq0$}\\0&\text{if $\nu<0$}\end{cases}$"
    ax.legend(title=lgd, loc=2, ncol=2)
    xlbl = r"$\log_{10}\left(\frac{L_\mathrm{IR}}{\mathrm{L}_\odot}\right)$"
    ylbl = r"$\log_{10}\left(\frac{L_\circledast}{\mathrm{L}_\odot}\right)$"

    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    relative_path_to_file = os.path.join(parent_dir, 'Data', 'fig3.jpg')
    fig.savefig(relative_path_to_file, dpi=600)
plt.show()