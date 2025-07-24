import os
import scipy.io
import numpy as np
import pandas as pd

#采用，分批处理，且最后只生成一个csv文件
def process_large_data(folder_path, output_csv, batch_size=100):
    # 首次写入时创建文件并写入header
    first_batch = True
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.mat'):
            filepath = os.path.join(folder_path, filename)
            mat_data = scipy.io.loadmat(filepath)
            receive_A = mat_data['receive_A']
            
            batch_data = []
            for col in range(receive_A.shape[1]):
                column_data = receive_A[:, col]
                for start_row in range(5200, 45200 - 4000 + 1, 2000):
                    segment = column_data[start_row:start_row+4000]
                    batch_data.append(segment)
                    
                    # 达到批次大小时写入文件
                    if len(batch_data) >= batch_size:
                        df = pd.DataFrame(batch_data)
                        df.to_csv(output_csv, mode='a', header=False, index=False)
                        batch_data = []
                        print(f"已写入批次到 {output_csv}")
            
            # 写入剩余数据
            if batch_data:
                df = pd.DataFrame(batch_data)
                df.to_csv(output_csv, mode='a', header=False, index=False)
                print(f"写入最后批次到 {output_csv}")

# 使用示例
process_large_data('D:\shiyan_data\cft', 'cft_processed_data.csv', batch_size=1000)