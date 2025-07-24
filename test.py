import os
import scipy.io
import numpy as np
import pandas as pd

#未使用，可参考，每处理一个.mat文件就生成一个csv文件
def process_mat_files(input_folder, output_folder, segment_length=100):
    """
    处理.mat文件中的receive_A矩阵，按列切割并保存为CSV
    
    参数:
        input_folder: 包含.mat文件的文件夹路径
        output_folder: 输出CSV文件的文件夹路径
        segment_length: 每个数据段的长度(默认为100)
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历输入文件夹中的所有.mat文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.mat'):
            # 构建完整文件路径
            mat_path = os.path.join(input_folder, filename)
            
            try:
                # 加载.mat文件
                mat_data = scipy.io.loadmat(mat_path)
                receive_A = mat_data['receive_A']
                
                # 检查矩阵维度
                if receive_A.shape[0] != 100000 or receive_A.shape[1] != 50:
                    print(f"警告: 文件 {filename} 中的receive_A维度不是100000x50，跳过处理")
                    continue
                
                # 准备存储所有切割后的数据
                all_segments = []
                
                # 对每一列进行处理
                for col in range(receive_A.shape[1]):
                    column_data = receive_A[:, col]
                    
                    # 计算可以切割的段数
                    num_segments = len(column_data) // segment_length
                    
                    # 切割数据
                    for i in range(num_segments):
                        start_idx = i * segment_length
                        end_idx = start_idx + segment_length
                        segment = column_data[start_idx:end_idx]
                        
                        # 添加列标签
                        segment_dict = {
                            f'Column_{col}_Segment_{i}': segment
                        }
                        all_segments.append(segment_dict)
                
                # 将数据转换为DataFrame
                # 这里需要将数据重新组织为适合CSV的格式
                # 创建一个DataFrame，每列是一个数据段
                df_data = {}
                for i, segment_dict in enumerate(all_segments):
                    for key, value in segment_dict.items():
                        df_data[key] = value
                
                df = pd.DataFrame(df_data)
                
                # 构建输出文件名
                base_name = os.path.splitext(filename)[0]
                csv_filename = f"{base_name}_segments.csv"
                csv_path = os.path.join(output_folder, csv_filename)
                
                # 保存为CSV
                df.to_csv(csv_path, index=False)
                print(f"已处理 {filename} 并保存为 {csv_filename}")
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

# 使用示例
input_folder = 'D:/shiyan_data/cft'  # 包含.mat文件的文件夹
output_folder = 'output_csv'  # 输出CSV文件的文件夹

process_mat_files(input_folder, output_folder)