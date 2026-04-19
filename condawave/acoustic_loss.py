import matplotlib.pyplot as plt
import numpy as np

# 设置新罗马字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 20

# ---------------- 模拟真实训练数据 ----------------
np.random.seed(42) 
epochs = np.arange(1, 51)

# 1. 生成 Training Accuracy (目标最终在 94% 左右)
# 调高了 base 值为 94.5
train_acc_base = 94.5 - 45.0 * np.exp(-0.15 * epochs)
# 训练集采用累积随机漫步，使其波动具有连续性，看起来更像真实收敛
train_noise = np.random.normal(0, 0.2, size=50).cumsum() * 0.12
train_acc = np.clip(train_acc_base + train_noise, 0, 99.5)

# 2. 生成 Validation Accuracy (目标在 92%~93% 之间，略低于训练集)
# 设置验证集上限略低于训练集，模拟轻微的泛化差距
val_acc_base = 92.8 - 47.0 * np.exp(-0.16 * epochs)
# 验证集增加一些随机扰动，模拟验证时的采样波动
val_noise = np.random.normal(0, 0.7, size=50) 
val_acc = np.clip(val_acc_base + val_noise, 0, 99.0)

# 3. 生成 Loss (准确率高，对应的 Loss 应该更低)
# 最终 Loss 稳定在 0.15 左右
loss_base = 0.15 + 2.8 * np.exp(-0.14 * epochs)
loss_noise = np.random.normal(0, 0.02, size=50)
loss = np.clip(loss_base + loss_noise, 0.05, 10)

# 转换为 list 并保留小数
epochs_list = epochs.tolist()
train_acc = [round(x, 2) for x in train_acc]
val_acc = [round(x, 2) for x in val_acc]
loss = [round(x, 4) for x in loss]

# ---------------- 开始绘图 ----------------
fig, ax1 = plt.subplots(figsize=(12, 7))

# 绘制准确率曲线
color1 = '#1f77b4'  # 蓝色
color2 = '#ff7f0e'  # 橙色
ax1.set_xlabel('Epoch', fontname='Times New Roman')
ax1.set_ylabel('Accuracy (%)', fontname='Times New Roman', color=color1)

line1 = ax1.plot(epochs_list, train_acc, color=color1, linestyle='-', marker='o', 
                 markersize=4, linewidth=2, label='Train Accuracy', alpha=0.9)
line2 = ax1.plot(epochs_list, val_acc, color=color2, linestyle='-', marker='s', 
                 markersize=4, linewidth=2, label='Validation Accuracy', alpha=0.9)

ax1.tick_params(axis='y', labelcolor=color1)
# 调整 Y 轴范围，使得 94% 左右的曲线处于中上方
ax1.set_ylim([40, 100])
ax1.grid(True, alpha=0.3, linestyle='--')

# 创建右y轴绘制loss
ax2 = ax1.twinx()
color3 = '#2ca02c'  # 绿色
ax2.set_ylabel('Loss', fontname='Times New Roman', color=color3)
line3 = ax2.plot(epochs_list, loss, color=color3, linestyle='--', marker='^', 
                 markersize=4, linewidth=1.5, label='Loss', alpha=0.7)
ax2.tick_params(axis='y', labelcolor=color3)
ax2.set_ylim([0, max(loss) * 1.1]) 

# 合并图例
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', frameon=True, shadow=True, fontsize=16)

# 设置x轴刻度
ax1.set_xticks(np.arange(0, 51, 5))

fig.tight_layout()
plt.show()

# 打印最终统计信息
print(f"最终训练准确率: {train_acc[-1]}%")
print(f"最终验证准确率: {val_acc[-1]}%")
print(f"最终损失值: {loss[-1]}")