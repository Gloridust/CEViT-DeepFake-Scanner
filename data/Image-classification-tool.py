import os
import shutil

# 可轻松更改的目录和文件路径变量
trainset_dir = './trainset'
label_file = 'trainset_label.txt'

# 创建结果目录
result_dir = './trainset_result'
fake_dir = os.path.join(result_dir, 'fake')
real_dir = os.path.join(result_dir, 'real')
os.makedirs(result_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)
os.makedirs(real_dir, exist_ok=True)

# 读取标签文件
with open(label_file, 'r') as f:
    lines = f.readlines()[1:]  # 跳过标题行

for line in lines:
    img_name, target = line.strip().split(',')
    src_img_path = os.path.join(trainset_dir, img_name)
    if target == '1':
        dst_img_path = os.path.join(fake_dir, img_name)
    else:
        dst_img_path = os.path.join(real_dir, img_name)
    if os.path.exists(src_img_path):
        shutil.move(src_img_path, dst_img_path)