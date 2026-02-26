import numpy as np
import matplotlib.pyplot as plt

# Confusion Matrix 数据
cm = np.array([
    [11253,  250,  272,  281,   85],
    [  761, 11319,    5,   13,   89],
    [  247,   31, 11160,  558,   56],
    [  217,   41,  175, 11744,  133],
    [  296,  234,  133,  482, 10965]
])

# 类别名称（可自行修改）
classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']

# 创建图像
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion Matrix', fontsize=14)
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# 在格子中显示数值
threshold = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

# 保存高清图像
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()