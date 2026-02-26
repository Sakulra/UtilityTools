import csv
import os
from tqdm import tqdm  # 用于显示进度条，如果没有安装可以通过 pip install tqdm 安装

def process_large_csv_with_labels(input_files, output_file):
    """
    处理多个大型CSV文件，为每个文件添加标签，并合并到一个输出文件中
    输出文件的第一行为列索引（1-4000为数据列，4001为标签列）
    
    参数:
        input_files: 包含文件路径和对应标签的列表，例如 [('file1.csv', 1), ('file2.csv', 2)]
        output_file: 合并后的输出文件路径
    """
    
    # 检查输出文件是否已存在，如果存在则删除
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # 打开输出文件准备写入
    with open(output_file, 'a', newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f)
        
        # 创建列索引（列名）：1到4000的数据列 + 标签列
        # 假设原始数据有4000列，所以列索引从1到4000，第4001列是标签
        total_data_columns = 4000
        column_headers = list(range(1, total_data_columns + 1)) + ['label']
        writer.writerow(column_headers)  # 写入列名行
        print(f"已添加列索引: 1-{total_data_columns} 和 'label'")
        
        # 处理每个输入文件
        for file_path, label in input_files:
            print(f"正在处理文件: {file_path} (标签: {label})")
            
            # 获取文件总行数用于进度条显示
            total_lines = 0
            with open(file_path, 'r', encoding='utf-8') as temp_f:
                total_lines = sum(1 for _ in temp_f)
            
            # 打开当前输入文件
            with open(file_path, 'r', encoding='utf-8') as in_f:
                reader = csv.reader(in_f)
                
                # 使用tqdm显示进度条
                for row in tqdm(reader, total=total_lines, desc=f"处理 {os.path.basename(file_path)}"):
                    # 检查数据列数是否符合预期
                    if len(row) != total_data_columns:
                        print(f"警告: 文件 {file_path} 中某行有 {len(row)} 列，预期是 {total_data_columns} 列")
                    
                    # 在每行数据末尾添加标签
                    row_with_label = row + [label]
                    writer.writerow(row_with_label)
    
    print(f"所有文件处理完成，结果已保存到: {output_file}")

# 示例使用
if __name__ == "__main__":
    # 定义输入文件和对应的标签
    input_files = [
        (r'D:\shiyan_data/剔除多余数据/cft_processed_4000.csv', 1),
        (r'D:\shiyan_data/剔除多余数据/qiu_processed_4000.csv', 2),
        (r'D:\shiyan_data/剔除多余数据/tuoyuan_processed_4000.csv', 3),
        (r'D:\shiyan_data/剔除多余数据/yuanzhu_processed_4000.csv', 4),
        (r'D:\shiyan_data/剔除多余数据/zft_processed_4000.csv', 5)
    ]
    
    # 输出文件路径
    output_file = 'add_labeled_4000.csv'
    
    # 调用处理函数
    process_large_csv_with_labels(input_files, output_file)