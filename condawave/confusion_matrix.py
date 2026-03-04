import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置新罗马字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体也使用类似新罗马的样式
matplotlib.rcParams['font.size'] = 16

# Confusion Matrix 数据
cm = np.array([
    [11253,  250,  272,  281,   85],
    [  761, 11319,    5,   13,   89],
    [  247,   31, 11160,  558,   56],
    [  217,   41,  175, 11744,  133],
    [  296,  234,  133,  482, 10965]
])

# 类别名称（可自行修改）
classes = ['cft', 'qiu', 'ty', 'yz', 'zft']

# 计算每个类别的总和（按行）
row_sums = cm.sum(axis=1)

# 创建图像
plt.figure(figsize=(10, 8))

# 使用蓝色系颜色映射
plt.imshow(cm, interpolation='nearest', cmap='Blues')
# plt.title('Confusion Matrix', fontsize=14)
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# 在格子中显示数值和百分比
threshold = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        # 计算百分比
        percentage = (cm[i, j] / row_sums[i]) * 100
        
        # 决定文本颜色
        text_color = "white" if cm[i, j] > threshold else "black"
        
        # 显示数值和百分比（数值换行，百分比加括号）
        plt.text(j, i, f'{cm[i, j]}\n({percentage:.1f}%)',
                 horizontalalignment="center",
                 verticalalignment="center",
                 color=text_color,
                 fontsize=9)

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

# 保存高清图像
plt.savefig("confusion_matrix_with_percentage.png", dpi=300)
plt.show()