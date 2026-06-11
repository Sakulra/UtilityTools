import scipy.io
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Times New Roman', "SimSun"]
plt.rcParams['font.size'] = 24

# 读取.mat文件
data = scipy.io.loadmat('F:/shiyan_data/无目标/18k2024年7月8日9时29分33.mat')
# data = scipy.io.loadmat('D:/shiyan_data/cft/18k2024年7月8日11时24分30.mat')#长方体
# data = scipy.io.loadmat('D:/shiyan_data/qiu/18k2024年7月8日20时24分45.mat')#球体
# data = scipy.io.loadmat('D:/shiyan_data/tuoyuan/18k2024年7月9日20时53分28.mat')#椭圆体
# data = scipy.io.loadmat('D:/shiyan_data/yuanzhu/18k2024年7月9日14时1分41.mat')#圆柱体
# data = scipy.io.loadmat('D:/shiyan_data/zft/16k2024年7月9日8时54分37.mat')#正方体



# 查看文件内容
print(data.keys())

# 提取声波数据
signal = data['receive_A'][:,0]
time = data["ts"][0,:]


# 处理声波数据
# mean_value = np.mean(signal)
# print(f"Mean value of the signal: {mean_value}")

# 可视化声波数据
plt.margins(x=0)
plt.plot(time,signal)
# plt.title('Sound Wave')
plt.xlabel('时间(s)')
plt.ylabel('幅度(V)')
plt.show()

# 保存处理后的数据
#scipy.io.savemat('processed_signal.mat', {'processed_signal': signal})