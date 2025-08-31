import pandas as pd

# # 读取CSV文件（没有列名）
# df = pd.read_csv(r'C:\Users\wice\Desktop\cft_processed_data.csv',header=None)

# # 在最后添加一列，赋值为1
# df['new_column'] = 1

# # 保存修改后的文件（可以选择覆盖原文件或保存为新文件）
# df.to_csv('your_file_with_new_column.csv', index=False, header=False)


def count_lines(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

# 使用
line_count = count_lines('F:\shiyan_data\processA/add_labeled_dataA.csv')
print(f"文件总行数: {line_count}")