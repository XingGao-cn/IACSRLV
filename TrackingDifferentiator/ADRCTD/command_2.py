# 提供原始指令
import numpy as np


# 获取单步指令
def get_command2(t, end):
    # alpha_c为攻角控制指令，beta_c为侧滑角控制指令，sigma_c为倾侧角控制指令
    t_1 = 20
    t_2 = 35
    t_3 = 60
    t_4 = 80

    alpha_1 = 18
    alpha_2 = 15
    alpha_3 = 10

    sigma_1 = 10
    sigma_2 = -15

    if t < t_2:
        alpha_c = alpha_1 * (t < t_1) + get_dot(alpha_1, alpha_2, t_1, end, t) * (t >= t_1 and t < t_2)
        beta_c = 0
        sigma_c = sigma_1 * (t < t_1) + get_dot(sigma_1, sigma_2, t_1, end, t) * (t >= t_1 and t < t_2)
    elif t < t_3 and t >= t_2:
        alpha_c = alpha_1 * (t < t_1) + get_dot(alpha_1, alpha_2, t_1, end, t) * (t >= t_2 and t < t_3)
        beta_c = 0
        sigma_c = sigma_1 * (t < t_1) + get_dot(sigma_1, sigma_2, t_1, end, t) * (t >= t_2 and t < t_3)
    elif t < t_4 and t >= t_3:
        alpha_c = alpha_2
        beta_c = 0
        sigma_c = sigma_1 * (t < t_1) + get_dot(sigma_1, sigma_2, t_1, end, t_3) * (t >= t_3 and t < t_4)
    else:
        alpha_c = alpha_2
        beta_c = 0
        sigma_c = 0

    aero_angle_d = np.array([alpha_c, beta_c, sigma_c])  # 姿态角指令 弧度制 单位/rad
    aero_angle_d_dot = np.array([0, 0, 0])  # 姿态角指令一阶导
    aero_angle_d_dot_dot = np.array([0, 0, 0])  # 姿态角指令二阶导

    X_d = np.concatenate((aero_angle_d, aero_angle_d_dot, aero_angle_d_dot_dot))  # shape: 1×9

    return X_d


def get_dot(angle1, angle2, t1, t2, t):
    # 斜坡姿态角控制指令计算
    # angle为姿态角控制指令输出
    # angle1为初始时刻姿态角，angle2为结束时刻姿态角
    # t1为初始时刻，t2为结束时刻，t为当前时刻

    angle = (angle2 - angle1) / (t2 - t1) * (t - t1) + angle1
    return angle


# 示例用法
# t_example = 35  # 请根据需要设置具体的时间t
# End_example = 150  # 请根据需要设置具体的End值
# result = get_command2(t_example, End_example)
# print(result)
# print(type(result[0]))
# print(getcontext().prec)
