from gymnasium.envs.registration import register
from ReinforcementLearning.gymnasium_demos.envs import RLVDynamicsEnv

import sys
import time
import numpy as np
from math import pi
import sys
import time

import matplotlib as mpl
from VehSimuParams.simulation_params import SimulationParameters
from matplotlib import pyplot as plt  # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
from pathlib import Path
import os

# 获取当前文件的绝对路径及父路径
module_path = os.path.dirname(os.path.realpath(__file__))
script_dir = Path(__file__).parent
parent_dir = script_dir.parent

# 将当前文件所在目录添加到 sys.path 中
sys.path.append(module_path)

# 注册自定义环境 (RLV)
# register(
#      id="gymnasium_demos/RLVDynamicsEnv-v0",  # id should then be used during environment creation
#      entry_point="gymnasium.envs:RLVDynamicsEnv",
#      max_episode_steps=500, # per episode
# )


if __name__ == "__main__":
    # 记录高精度的时间开销
    start_time = time.perf_counter()

    # 打印运行代码时间
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Simulation executed in {execution_time} seconds with high precision.")
