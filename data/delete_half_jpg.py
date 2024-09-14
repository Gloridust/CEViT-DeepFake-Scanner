import os
import random

def delete_half_jpg_files(directory):
    # 获取指定目录下所有的 .jpg 文件
    jpg_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]

    # 确定需要删除的文件数量（随机删除一半）
    num_files_to_delete = len(jpg_files) // 2

    # 随机选择需要删除的文件
    files_to_delete = random.sample(jpg_files, num_files_to_delete)

    # 删除选中的文件
    for file in files_to_delete:
        file_path = os.path.join(directory, file)
        os.remove(file_path)
        print(f"Deleted: {file_path}")

    print(f"Total {num_files_to_delete} files deleted.")

# 使用示例：指定目录路径
directory = "./data_min/train/fake/"  # 替换为你想指定的目录路径
delete_half_jpg_files(directory)