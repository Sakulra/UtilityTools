import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置新罗马字体
plt.rcParams['font.family'] = ['Times New Roman',"SimSun"]
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体也使用类似新罗马的样式
matplotlib.rcParams['font.size'] = 18

# Confusion Matrix 数据
#训练集共608000 行数据
# cm = np.array([
#     [112530,  2500,  2720,  2810,   850],
#     [  7610, 113190,    50,   130,   890],
#     [  2470,   310, 111600,  5580,   560],
#     [  2170,   410,  1750, 117440,  1330],
#     [  2960,  2340,  1330,  4820, 109650]
# ])
#测试集152000 行数据
cm = np.array([
    [28012, 566, 641, 768, 198],
    [1851, 28343, 4, 36, 215],
    [634, 56, 28397 , 1309 , 131],
    [537,   112,   422, 29039, 374],
    [722,   528,   313 , 1182 ,27610]])

# 类别名称（可自行修改）
classes = ['长方体', '球体', '椭圆体', '圆柱体', '正方体']

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
                 fontsize=18)

plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.tight_layout()

# 保存高清图像
plt.savefig("confusion_matrix_with_percentage.png", dpi=300)
plt.show()