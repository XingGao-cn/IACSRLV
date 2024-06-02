# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

"""
   - import 需要给出完整路径（from x.xx import xxx ）
   - 使用np.float64以保证仿真精度
   - 使用math库替代np.math以保证运行效率
   - 矩阵乘法用np.dot, 点积用np.multiply, 数值乘法用*, 多个矩阵连乘用@
   - 类名使用驼峰命名法
   - py文件名、函数名、变量名使用下划线命名法
   - 常量使用大写+_命名法
   - python数据结构下标从0开始
   - 导入类代替导入类所在文件以获取类内部变量的访问权限
   - np.array列/行向量定义后、使用前需reshape以支持索引; 此外, 索引时应同时指定行列
   - numpy中一维数组与二维数组存在区别
   - 切片索引[m,n)
   - 对于一维向量, 统一使用行向量; 对于返回值, 存在列向量 (控制量)
   - 禁止在重复调用的函数中创建同一类变量
   - 逐行对照代码, 逐块测试输出
"""

from Cmd import initialization


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# main
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('This is the main file')

    # Initialize
    initialization



# Simulation process
# while True:
#



# Plot results (you need to implement the figure_plot function)
# figure_plot()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
