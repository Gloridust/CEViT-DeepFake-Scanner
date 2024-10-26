# dataset.py
# 定义了数据集的加载和预处理

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class FaceDataset(Dataset):
    def __init__(self, data_dir, train=True, target_size=(384, 384)):
        self.data_dir = data_dir
        self.train = train
        self.target_size = target_size
        self.image_paths = []
        self.labels = []

        # 支持的图像文件扩展名
        valid_image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

        real_dir = os.path.join(data_dir, 'real')
        fake_dir = os.path.join(data_dir, 'fake')

        # 只读取合法的图片文件
        for img_name in os.listdir(real_dir):
            if img_name.lower().endswith(valid_image_extensions):
                self.image_paths.append(os.path.join(real_dir, img_name))
                self.labels.append(0)  # 真实人脸标签为0

        for img_name in os.listdir(fake_dir):
            if img_name.lower().endswith(valid_image_extensions):
                self.image_paths.append(os.path.join(fake_dir, img_name))
                self.labels.append(1)  # AI生成人脸标签为1

        # 定义基本变换
        self.basic_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 为训练集添加数据增强
        if train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.RandomVerticalFlip(),  # 随机垂直翻转
                transforms.RandomRotation(degrees=15),  # 随机旋转，旋转角度为15度
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 颜色抖动，调整亮度、对比度、饱和度和色调
                transforms.RandomResizedCrop(self.target_size, scale=(0.8, 1.0)),  # 随机裁剪并调整大小，裁剪比例在0.8到1.0之间
                transforms.RandomGrayscale(p=0.1),  # 随机灰度转换，概率为0.1
                # 新增以下四种增强方法
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),  # 随机锐化，锐化因子为2，概率为0.5
                transforms.RandomAutocontrast(p=0.5),  # 随机自动对比度，概率为0.5
                transforms.RandomEqualize(p=0.5),  # 随机均衡化，概率为0.5
                transforms.RandomPerspective(distortion_scale=0.3, p=0.5),  # 透视变换，失真比例为0.3，概率为0.5
                self.basic_transform  # 基本变换
            ])
        else:
            self.transform = self.basic_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 打开图像并确保它是RGB模式
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        image = self.transform(image)

        # 确保标签为 long 类型的张量
        label = torch.tensor(label, dtype=torch.long)

        return image, label
