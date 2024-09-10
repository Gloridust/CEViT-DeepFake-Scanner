import os
import shutil

# 可轻松更改的目录和文件路径
trainset_folder = './trainset'
label_file = 'trainset_label.txt'

result_folder = './result/trainset'
fake_folder = os.path.join(result_folder, 'fake')
real_folder = os.path.join(result_folder, 'real')

# 如果结果文件夹不存在则创建
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
if not os.path.exists(fake_folder):
    os.makedirs(fake_folder)
if not os.path.exists(real_folder):
    os.makedirs(real_folder)

# 读取标签文件
with open(label_file, 'r') as f:
    lines = f.readlines()[1:]  # 跳过标题行

for line in lines:
    parts = line.strip().split(',')
    img_name = parts[0]
    target = int(parts[1])
    src_path = os.path.join(trainset_folder, img_name)
    if target == 1:
        dst_path = os.path.join(fake_folder, img_name)
    else:
        dst_path = os.path.join(real_folder, img_name)
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)