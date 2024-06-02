import numpy as np

# 创建一个形状为(1, 7)的数组
original_array = np.ones((1, 7))

# 使用reshape方法将其变为(7,)
reshaped_array = original_array.reshape((-1,7))

print(reshaped_array.shape)  # 输出将会是 (7,)