import os
import scipy.io
import numpy as np
import pandas as pd

def process_mat_files(folder_path, output_csv):
    # 初始化一个空列表来存储所有处理后的数据
    all_data = []
    
    # 遍历文件夹中的所有.mat文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.mat'):
            filepath = os.path.join(folder_path, filename)
            
            # 加载.mat文件
            mat_data = scipy.io.loadmat(filepath)
            receive_A = mat_data['receive_A']
            
            # 处理每一列
            for col in range(receive_A.shape[1]):  # 遍历50列
                column_data = receive_A[:, col]
                
                # 提取5200到45200行，步长2000
                for start_row in range(5200, 45200 - 4000 + 1, 2000):
                    end_row = start_row + 4000
                    segment = column_data[start_row:end_row]
                    
                    # 将段添加到所有数据中
                    all_data.append(segment)
    
    # 将列表转换为numpy数组
    all_data_array = np.array(all_data)
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(all_data_array)
    df.to_csv(output_csv, index=False, header=False)
    print(f"处理完成，数据已保存到 {output_csv}")

# 使用示例
testdata_folder = 'D:\shiyan_data\cft'  # 替换为您的文件夹路径
output_csv = 'cft_processed_data.csv'  # 输出文件名
process_mat_files(testdata_folder, output_csv)