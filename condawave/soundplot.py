import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# 读取.mat文件
data = scipy.io.loadmat('C:/Users/001\Desktop/12k2024年7月8日9时25分25.mat')

# 查看文件内容
print(data.keys())

# 提取声波数据
signal = data['receive_A'][:,0]
time = data["ts"][0,:]


# 处理声波数据
# mean_value = np.mean(signal)
# print(f"Mean value of the signal: {mean_value}")

# 可视化声波数据
plt.plot(time,signal)
plt.title('Sound Wave')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# 保存处理后的数据
#scipy.io.savemat('processed_signal.mat', {'processed_signal': signal})