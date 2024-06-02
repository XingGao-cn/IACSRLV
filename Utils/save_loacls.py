import pickle
import sys
import os

# 获取当前工作区的所有局部变量
local_vars = locals()


# 定义一个函数来检查对象是否可以被pickle
def is_picklable(obj):
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, TypeError):
        return False


# 过滤掉不可序列化的对象，包括模块对象
to_save = {k: v for k, v in local_vars.items() if not k.startswith('__') and is_picklable(v)}


# 定义保存变量的函数
def save_all_vars_to_file(filename='saved_vars.pickle', vars_dict=to_save):
    with open(filename, 'wb') as file:
        pickle.dump(vars_dict, file)


a = 3
b = 4
# 调用函数保存变量
save_all_vars_to_file()


# 之后，您可以从文件中加载这些变量
def load_all_vars_from_file(filename='saved_vars.pickle'):
    with open(filename, 'rb') as file:
        vars_dict = pickle.load(file)
    return vars_dict


# 加载变量
loaded_vars = load_all_vars_from_file()

# 打印加载的变量，确认它们已正确恢复
for key, value in loaded_vars.items():
    print(f'Loaded {key}: {value}')
