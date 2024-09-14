import os
import random

# 获取当前目录下所有的 .jpg 文件
jpg_files = [f for f in os.listdir() if f.endswith('.jpg')]

# 确定需要删除的文件数量（随机删除一半）
num_files_to_delete = len(jpg_files) // 2

# 随机选择需要删除的文件
files_to_delete = random.sample(jpg_files, num_files_to_delete)

# 删除选中的文件
for file in files_to_delete:
    os.remove(file)
    print(f"Deleted: {file}")

print(f"Total {num_files_to_delete} files deleted.")