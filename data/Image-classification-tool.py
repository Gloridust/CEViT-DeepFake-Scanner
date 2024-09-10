import os
import shutil

# 定义目录和文件路径变量，可以根据需要轻松更改
trainset_folder = './trainset'
label_file = 'trainset_label.txt'

# 读取标签文件
with open(label_file, 'r') as f:
    lines = f.readlines()[1:]  # 跳过标题行

# 遍历每一行标签数据
for line in lines:
    parts = line.strip().split(',')
    img_name = parts[0]
    target = int(parts[1])
    src_path = os.path.join(trainset_folder, img_name)
    if target == 1:
        dst_folder = './fake/'
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        dst_path = os.path.join(dst_folder, img_name)
        shutil.move(src_path, dst_path)
    elif target == 0:
        dst_folder = './real/'
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        dst_path = os.path.join(dst_folder, img_name)
        shutil.move(src_path, dst_path)