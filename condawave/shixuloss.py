import matplotlib.pyplot as plt
import numpy as np

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

train_acc = [35.91, 52.00, 63.00, 68.94, 72.51, 75.11, 77.23, 77.65, 78.25, 80.33, 
             81.55, 82.46, 83.25, 83.88, 84.51, 84.90, 85.43, 85.75, 86.12, 86.46,
             86.74, 87.03, 87.14, 87.37, 87.58, 87.68, 87.98, 88.14, 88.24, 88.43,
             88.53, 88.59, 88.75, 88.84, 88.97, 89.05, 89.16, 89.29, 89.38, 89.47,
             89.62, 89.67, 89.77, 89.76, 89.97, 89.94, 90.15, 90.14, 90.23, 90.24]

val_acc = [48.70, 60.47, 68.52, 73.12, 75.57, 80.26, 80.88, 79.92, 81.89, 83.80,
           85.01, 85.18, 86.57, 86.37, 87.53, 88.76, 87.82, 88.79, 89.13, 88.13,
           90.10, 88.36, 90.33, 89.74, 90.33, 89.45, 90.55, 89.19, 89.66, 89.95,
           90.68, 90.51, 90.94, 90.44, 91.98, 91.24, 91.33, 91.15, 91.39, 91.54,
           90.95, 92.52, 91.12, 92.22, 92.71, 91.44, 92.55, 91.73, 92.03, 92.83]

loss = [1.6199, 1.3294, 1.0770, 0.8223, 0.6147, 0.7172, 0.6231, 0.5917, 0.5164, 0.5397,
        0.4936, 0.4621, 0.4518, 0.5613, 0.4140, 0.4022, 0.5001, 0.3587, 0.4602, 0.3561,
        0.1837, 0.2820, 0.2248, 0.2058, 0.1734, 0.3047, 0.2168, 0.3022, 0.2183, 0.3676,
        0.1846, 0.2369, 0.2541, 0.1684, 0.3519, 0.2645, 0.2411, 0.3267, 0.2644, 0.2482,
        0.2086, 0.1776, 0.2853, 0.3301, 0.2971, 0.1598, 0.2583, 0.2414, 0.3099, 0.2559]

epochs = np.arange(1, 51)

fig, ax1 = plt.subplots(figsize=(10, 6))

# 左轴：Accuracy
ax1.plot(epochs, train_acc, marker='o', linestyle='-', label='Train Acc')
ax1.plot(epochs, val_acc, marker='s', linestyle='-', label='Val Acc')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)')
ax1.set_xlim(1, 50)
ax1.grid(True)

# 右轴：Loss
ax2 = ax1.twinx()
ax2.plot(epochs, loss, marker='^', linestyle='--', label='Loss')
ax2.set_ylabel('Loss')

# 合并图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

# plt.title('Training and Validation Metrics')
plt.tight_layout()
plt.show()