import csv
import os
from tqdm import tqdm  # 用于显示进度条，如果没有安装可以通过 pip install tqdm 安装

#由于文件很大，将使用逐行处理的方式以避免内存问题
def process_large_csv_with_labels(input_files, output_file):
    """
    处理多个大型CSV文件，为每个文件添加标签，并合并到一个输出文件中
    
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
                    # 在每行数据末尾添加标签
                    row_with_label = row + [label]
                    writer.writerow(row_with_label)
    
    print(f"所有文件处理完成，结果已保存到: {output_file}")

# 示例使用
if __name__ == "__main__":
    # 定义输入文件和对应的标签
    input_files = [
        ('E:\shiyan_data\processA\cft_processed_data.csv', 1),
        ('E:\shiyan_data\processA\qiu_processed_data.csv', 2),
        ('E:\shiyan_data\processA\\tuoyuan_processed_data.csv', 3),
        ('E:\shiyan_data\processA\yuanzhu_processed_data.csv', 4),
        ('E:\shiyan_data\processA\zft_processed_data.csv', 5)
    ]
    
    # 输出文件路径
    output_file = 'add_labeled_dataA.csv'
    
    # 调用处理函数
    process_large_csv_with_labels(input_files, output_file)