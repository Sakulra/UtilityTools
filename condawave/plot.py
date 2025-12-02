#读取csv前四行数据并画在一起
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件的前四行
try:
    # 请将'your_file.csv'替换为你的实际文件名
    df = pd.read_csv('E:\shiyan_data\processA\cft_processed_data.csv', nrows=4)
    
    # 创建2x2的子图画布
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Divided part of the data', fontsize=16)
    
    # 假设每行数据都是数值型数据点
    for i, (index, row) in enumerate(df.iterrows()):
        # 确定子图位置
        ax = axes[i//2, i%2]
        
        # 绘制离散点图
        ax.scatter(range(len(row)), row.values, color='blue', alpha=0.7, s=5)
        
        # 设置标题和标签
        # ax.set_title(f'分割后的部分数据')
        ax.set_xlabel('index')
        ax.set_ylabel('value')
        
        # 添加网格以便更好地读取数值
        ax.grid(True, alpha=0.3)
        
        # 如果数据点太多，可以调整x轴显示
        if len(row) > 20:
            ax.set_xticks(np.arange(0, len(row), max(1, len(row)//10)))
    
    # 调整布局
    plt.tight_layout()
    plt.show()
    
except FileNotFoundError:
    print("文件未找到，请检查文件名和路径是否正确")
except Exception as e:
    print(f"发生错误: {e}")

# 同时显示前四行数据的内容
print("前四行数据的内容：")
print(df.head(4))