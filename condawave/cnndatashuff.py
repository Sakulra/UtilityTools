#将添加好标签有序的总csv文件打乱并分为训练集和测试集

import csv
import random
import os
import numpy as np
from tqdm import tqdm

def split_large_csv(input_file, output_dir, train_ratio=0.8):
    """
    处理大文件的分割版本（内存友好，分块处理）
    
    参数:
        input_file: 输入文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 首先读取表头
    with open(input_file, 'r', encoding='utf-8') as f:
        header = next(csv.reader([f.readline()]))
    
    # 计算总行数（不包括表头）
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f) - 1
    
    print(f"总数据行数: {total_lines}")
    
    # 计算训练集行数
    train_size = int(total_lines * train_ratio)
    
    # 生成随机索引顺序
    indices = list(range(total_lines))
    random.shuffle(indices)
    train_indices = set(indices[:train_size])
    
    print(f"训练集: {train_size} 行, 测试集: {total_lines - train_size} 行")
    
    # 打开输出文件
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    with open(train_path, 'w', newline='', encoding='utf-8') as train_f, \
         open(test_path, 'w', newline='', encoding='utf-8') as test_f:
        
        train_writer = csv.writer(train_f)
        test_writer = csv.writer(test_f)
        
        # 写入表头
        train_writer.writerow(header)
        test_writer.writerow(header)
        
        # 逐行读取并根据随机索引分配
        with open(input_file, 'r', encoding='utf-8') as f:
            next(f)  # 跳过表头
            reader = csv.reader(f)
            
            for i, row in enumerate(tqdm(reader, total=total_lines, desc="处理数据")):
                if i in train_indices:
                    train_writer.writerow(row)
                else:
                    test_writer.writerow(row)
    
    print(f"处理完成!")
    print(f"训练集: {train_path}")
    print(f"测试集: {test_path}")

# 如果文件很大，使用这个版本
split_large_csv('./add_labeled_4000.csv', 'shuffled_dataset', train_ratio=0.8)