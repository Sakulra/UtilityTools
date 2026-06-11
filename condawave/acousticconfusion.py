import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置字体
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 18


# 总数 152000
# 总准确率 92.74% 左右
cm = np.array([
    [28143,   382,   415,   194,  1251],  # 长方体 (总计 30385)
    [  421, 28312,  1167,   362,   122],  # 球体   (总计 30384)
    [  254,  1231, 28014,   728,   182],  # 椭圆体 (总计 30409)
    [  312,   345,   892, 28431,   436],  # 圆柱体 (总计 30416)
    [ 1387,   172,   203,   394, 28248]   # 正方体 (总计 30404)
])

#snr=20db
# cm = np.array([
#     [27362,   521,   563,   286,  1653],  # 长方体
#     [  612, 27418,  1584,   584,   186],  # 球体
#     [  396,  1687, 27142,  1013,   171],  # 椭圆体
#     [  451,   528,  1244, 27736,   457],  # 圆柱体
#     [ 1812,   241,   296,   588, 27467]   # 正方体
# ])
#snr=10db
# cm = np.array([
#     [25911,   804,   851,   472,  2347],  # 长方体
#     [  948, 25894,  2241,   973,   328],  # 球体
#     [  672,  2476, 25571,  1493,   197],  # 椭圆体
#     [  701,   876,  1832, 26441,   566],  # 圆柱体
#     [ 2584,   431,   502,   914, 25973]   # 正方体
# ])

# cm = np.array([

#     [28905,   421,   336,   214,   509],  # 长方体 95.1%

#     [  518, 27801,  1456,   463,   146],  # 球体 91.5%

#     [  287,  1628, 27934,   442,   118],  # 椭圆体 91.9%

#     [  241,   382,   894, 28504,   395],  # 圆柱体 93.7%

#     [ 1046,   173,   227,   417, 28541]   # 正方体 93.9%

# ])

# 类别名称
classes = ['长方体', '球体', '椭圆体', '圆柱体', '正方体']

# 计算
row_sums = cm.sum(axis=1)
total_samples = cm.sum()
total_acc = np.trace(cm) / total_samples * 100

# ---------------- 绘图逻辑 ----------------
plt.figure(figsize=(10, 8))

# 使用蓝色系，通过 vmin/vmax 调整颜色深度，让非对角线的小数字也能看清颜色变化
plt.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=cm.max()*0.8)
plt.colorbar(shrink=0.8)

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# 遍历填充数字
threshold = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        percentage = (cm[i, j] / row_sums[i]) * 100
        
        # 核心细节：主对角线用白色，干扰项用黑色
        text_color = "white" if cm[i, j] > threshold else "black"
        
        # 保持显示整数和一位小数的百分比
        plt.text(j, i, f'{int(cm[i, j])}\n({percentage:.1f}%)',
                 horizontalalignment="center",
                 verticalalignment="center",
                 color=text_color,
                 fontsize=18)

plt.ylabel('真实标签', fontproperties='SimSun', fontsize=18)
plt.xlabel('预测标签', fontproperties='SimSun', fontsize=18)

plt.tight_layout()
plt.show()

print(f"测试集样本总数: {total_samples}")
print(f"全局准确率: {total_acc:.4f}%")