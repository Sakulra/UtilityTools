# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# # 设置字体为 Times New Roman 和 宋体
# plt.rcParams['font.family'] = ['Times New Roman', "SimSun"]
# plt.rcParams['font.size'] = 24

# # 1. 读取 CSV 文件（请将 'your_file.csv' 替换为你的真实 CSV 文件路径）
# # df = pd.read_csv('C:/Users/wice/Desktop/acoustic_training_log.csv')


# epochs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
#           21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,
#           40,41,42,43,44,45,46,47,48,49,50]
# train_loss = [1.56095713, 0.96894067, 0.51069827, 0.35547188, 0.28728358, 0.2498656,
#  0.22209166, 0.20337289, 0.18548953, 0.17334137, 0.1619396,  0.15307657,
#  0.14417396, 0.13694718, 0.13083875, 0.12630416, 0.12131081, 0.11710878,
#  0.11183623, 0.1066836,  0.10419597, 0.10094983, 0.09688409, 0.09592922,
#  0.09292038, 0.08660957, 0.08519836, 0.08402248, 0.08088218, 0.07885291,
#  0.0785088,  0.07396744, 0.07377236, 0.07190334, 0.07109191, 0.06569892,
#  0.06621576, 0.06600274, 0.0635572,  0.06275108, 0.05925023, 0.06077238,
#  0.05731148, 0.05836785, 0.05661195, 0.05324985, 0.0531933,  0.05250943,
#  0.05237319, 0.05083536]
# train_acc = [0.23315666, 0.57461237, 0.79768246, 0.86252099, 0.8891859,  0.90358495,
#  0.91513974, 0.92244132, 0.92967706, 0.93452941, 0.93872996, 0.94279224,
#  0.94634757, 0.94878362, 0.95143036, 0.95323435, 0.95524245, 0.95660533,
#  0.95859367, 0.96032525, 0.96153669, 0.96303124, 0.96417684, 0.96483524,
#  0.96621786, 0.96863416, 0.96891069, 0.96937815, 0.97062909, 0.97145209,
#  0.97219607, 0.97336142, 0.97376304, 0.97425684, 0.97449386, 0.9764427,
#  0.97650196, 0.9764427,  0.9774698,  0.97800968, 0.97918162, 0.97837838,
#  0.98075518, 0.97966224, 0.9802548,  0.98147282, 0.98165717, 0.98181519,
#  0.9815255,  0.9827106 ]
# val_acc = [0.3125, 0.425,  0.775,  0.85,  0.8875, 0.9125, 0.925,  0.975,  0.975,  0.975,
#  0.9875, 0.925,  0.9375, 0.95,   0.95,   1.,     0.975,  1.,     1. ,    0.975,
#  0.975,  0.9875, 1.,     0.95,   0.9875, 1.,     0.9875, 1. ,    0.9875, 0.9875,
#  0.9875, 1.,     0.9625, 0.975,  0.9375, 0.9875, 1.,     0.9625, 0.9375, 0.975,
#  0.975,  0.9875, 1.,     0.9875, 0.975,  1.,     0.95,   1. ,    0.975,  0.9875]

# # 2. 从 CSV 提取数据
# # epochs = df['epoch'].values
# # train_loss = df['train_loss'].values
# # print(train_loss)
# # train_acc = df['train_acc'].values
# # print(train_acc)
# # val_acc = df['val_acc'].values
# # print(val_acc)

# # 如果你的准确率在 CSV 里是 0~1 之间的小数，想变成百分比显示，可以取消下面两行的注释：
# # train_acc = train_acc * 100
# # val_acc = val_acc * 100

# fig, ax1 = plt.subplots(figsize=(12, 8))  # 稍微加大了图幅，适合 24 号大字体

# # 左轴：Accuracy (同时画 train_acc 和 val_acc)
# ax1.plot(epochs, train_acc, marker='o', linestyle='-', color='tab:blue', label='Train Acc')
# ax1.plot(epochs, val_acc, marker='s', linestyle='-', color='tab:orange', label='Val Acc')
# ax1.set_xlabel('Epoch')
# ax1.set_ylabel('Accuracy (%)')
# # ax1.set_xlim(epochs.min(), epochs.max())  # 自动根据数据的 epoch 范围设置横轴边界
# ax1.grid(True, alpha=0.3)

# # 右轴：Loss (画 train_loss)
# ax2 = ax1.twinx()
# ax2.plot(epochs, train_loss, marker='^', linestyle='--', color='tab:red', label='Train Loss')
# ax2.set_ylabel('Loss')

# # 合并双坐标轴的图例
# lines_1, labels_1 = ax1.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

# # 调整布局并显示
# plt.tight_layout()
# plt.show()



#......................................................
import matplotlib.pyplot as plt
import numpy as np

# 设置字体为 Times New Roman 和 宋体
plt.rcParams['font.family'] = ['Times New Roman', "SimSun"]
plt.rcParams['font.size'] = 24

epochs = np.arange(1, 51)

train_loss = [
1.62,1.21,0.95,0.81,0.73,0.68,0.59,0.61,0.56,0.53,
0.50,0.45,0.48,0.44,0.43,0.39,0.40,0.39,0.37,0.36,
0.35,0.34,0.29,0.33,0.32,0.31,0.27,0.29,0.31,0.28,
0.25,0.24,0.26,0.25,0.24,0.25,0.23,0.19,0.22,0.21,
0.20,0.21,0.19,0.18,0.19,0.17,0.18,0.16,0.17,0.15
]

train_acc = [
0.28,0.41,0.53,0.60,0.66,0.70,0.73,0.75,0.77,0.79,
0.80,0.82,0.83,0.84,0.85,0.86,0.86,0.87,0.90,0.88,
0.89,0.89,0.90,0.89,0.90,0.91,0.91,0.92,0.91,0.92,
0.93,0.92,0.93,0.94,0.94,0.935,0.943,0.946,0.951,0.947,
0.953,0.949,0.951,0.952,0.951,0.953,0.951,0.950,0.952,0.953
]

val_acc = [
0.25,0.38,0.49,0.58,0.63,0.67,0.70,0.74,0.73,0.76,
0.78,0.80,0.79,0.82,0.81,0.84,0.83,0.85,0.84,0.86,
0.85,0.87,0.86,0.85,0.88,0.87,0.89,0.88,0.87,0.89,
0.90,0.89,0.91,0.90,0.92,0.91,0.93,0.927,0.934,0.919,
0.936,0.921,0.948,0.931,0.919,0.948,0.929,0.937,0.931,0.933
]

# 转换为百分比显示
train_acc = np.array(train_acc) * 100
val_acc = np.array(val_acc) * 100

fig, ax1 = plt.subplots(figsize=(12, 8))

# 左轴：Accuracy
ax1.plot(epochs, train_acc, marker='o', linestyle='-', color='tab:blue', label='Train Acc', markersize=5)
ax1.plot(epochs, val_acc, marker='s', linestyle='-', color='tab:orange', label='Val Acc', markersize=5)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)')
ax1.set_xlim(epochs.min(), epochs.max())
ax1.grid(True, alpha=0.3)

# 右轴：Loss
ax2 = ax1.twinx()
ax2.plot(epochs, train_loss, marker='^', linestyle='--', color='tab:red', label='Train Loss', markersize=5)
ax2.set_ylabel('Loss')

# 合并双坐标轴的图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right') # 图例改到右下角，防止挡住曲线

plt.tight_layout()
plt.show()