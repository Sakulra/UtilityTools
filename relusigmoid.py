import numpy as np
import matplotlib.pyplot as plt

# 设置字体为 Times New Roman，字号18
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

# 定义 x 范围
x = np.linspace(-10, 10, 1000)

# 定义函数
relu = np.maximum(0, x)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)

# 创建图像
plt.figure(figsize=(8, 6))

# 绘制函数
plt.plot(x, relu, label='ReLU', linewidth=2)
plt.plot(x, sigmoid, label='Sigmoid', linewidth=2)
plt.plot(x, tanh, label='Tanh', linewidth=2)

# 显示坐标轴
plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)

# 添加图例
plt.legend()

# 添加标题
# plt.title('Activation Functions')

# 显示网格（可选）
plt.grid(True)

# 显示图像
plt.show()