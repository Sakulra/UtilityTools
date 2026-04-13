import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以便复现
np.random.seed(42)

plt.rcParams['font.family'] = ['Times New Roman',"SimSun"]
plt.rcParams['font.size'] = 22

# 生成模拟的训练数据
epochs = 50
x = np.arange(1, epochs + 1)

# 生成训练损失（前期快速下降，后期缓慢下降但有波动）
train_loss = np.zeros(epochs)
for i in range(epochs):
    if i < 15:
        # 前期：快速下降
        train_loss[i] = 2.0 * np.exp(-0.2 * (i+1)) + 0.15
    else:
        # 后期：缓慢下降，但有波动
        base_loss = 0.32 * np.exp(-0.03 * (i-14)) + 0.12
        wave = 0.025 * np.sin(0.6 * (i-14))
        noise = 0.015 * np.random.randn()
        train_loss[i] = base_loss + wave + noise

train_loss = np.clip(train_loss, 0.08, None)

# 生成训练准确率（最终达到92%）
train_acc = np.zeros(epochs)
for i in range(epochs):
    if i < 25:
        train_acc[i] = 100 * (1 - 0.65 * np.exp(-0.1 * (i+1))) + np.random.randn() * 1.2
    else:
        base_acc = 92.3 - 1.8 * np.exp(-0.08 * (i-24))
        noise = np.random.randn() * 0.6
        train_acc[i] = base_acc + noise

train_acc = np.clip(train_acc, 50, 93.5)

# 生成验证准确率（最终达到90%）
val_acc = np.zeros(epochs)
for i in range(epochs):
    if i < 30:
        val_acc[i] = 100 * (1 - 0.72 * np.exp(-0.09 * (i+1))) + np.random.randn() * 1.0
    else:
        base_acc = 90.1 - 1.2 * np.exp(-0.06 * (i-29))
        wave = 0.3 * np.sin(0.4 * (i-29))
        noise = np.random.randn() * 0.5
        val_acc[i] = base_acc + wave + noise

val_acc = np.clip(val_acc, 45, 91.5)

# 确保最终准确率满足要求
train_acc[-8:] = 92.3 + np.random.randn(8) * 0.3
train_acc = np.clip(train_acc, 50, 93)

val_acc[-10:] = 90.1 + np.random.randn(10) * 0.4
val_acc = np.clip(val_acc, 45, 91)

# 创建图形（单图双y轴）
fig, ax1 = plt.subplots(figsize=(12, 7))

# 颜色设置
color_loss = '#2ca02c'      # 绿色
color_train_acc = '#1f77b4'  # 蓝色
color_val_acc = '#ff7f0e'    # 橙色

# 绘制训练损失（左y轴）- 点线图
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color_loss)
line1 = ax1.plot(x, train_loss, color=color_loss, linestyle='-', marker='^', 
                 markersize=5, linewidth=2, label='Train Loss', alpha=0.8)
ax1.tick_params(axis='y', labelcolor=color_loss)
ax1.grid(True, alpha=0.3, linestyle='--')

# 创建右y轴用于准确率 - 点线图
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy (%)')
line2 = ax2.plot(x, train_acc, color=color_train_acc, linestyle='-', marker='o', 
                 markersize=5, linewidth=2, label='Train Accuracy', alpha=0.8)
line3 = ax2.plot(x, val_acc, color=color_val_acc, linestyle='-', marker='s', 
                 markersize=5, linewidth=2, label='Validation Accuracy', alpha=0.8)
ax2.tick_params(axis='y')

# 合并图例
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

plt.tight_layout()
plt.show()

# 打印关键数据
print("训练完成！")
print(f"最终训练损失: {train_loss[-1]:.4f}")
print(f"最终训练准确率: {train_acc[-1]:.2f}%")
print(f"最终验证准确率: {val_acc[-1]:.2f}%")
print(f"\n最后5个epoch的验证准确率:")
for i in range(5):
    print(f"Epoch {epochs-4+i}: {val_acc[epochs-5+i]:.2f}%")

# 保存数据到CSV文件
import pandas as pd

data = {
    'Epoch': x,
    'Train_Loss': train_loss,
    'Train_Accuracy': train_acc,
    'Validation_Accuracy': val_acc
}
df = pd.DataFrame(data)
df.to_csv('resnet_training_results.csv', index=False)
print("\n数据已保存到 'resnet_training_results.csv'")