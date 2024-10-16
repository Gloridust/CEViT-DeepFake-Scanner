# dataset.py
# 定义了数据集的加载和预处理

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np

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

        # 为训练集添加更多数据增强
        if train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(self.target_size, scale=(0.8, 1.0)),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomApply([
                    transforms.Lambda(self.mixup)
                ], p=0.5),
                transforms.RandomApply([
                    transforms.Lambda(self.cutmix)
                ], p=0.5),
                self.basic_transform
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
        
        image = self.transform(image)  # 先应用变换，确保图像尺寸一致

        if self.train:
            if np.random.rand() < 0.5:
                image, label = self.mixup(image, label)  # mixup 操作现在在变换之后，确保图像尺寸一致
            elif np.random.rand() < 0.5:
                image, label = self.cutmix(image, label)

        # 确保标签为 float32 类型的张量
        label = torch.tensor(label, dtype=torch.float32)

        return image, label

    # 定义 MixUp 增强
    def mixup(self, image, label):
        # 随机选择一个样本
        rand_idx = np.random.randint(0, len(self.image_paths))
        mix_img_path = self.image_paths[rand_idx]
        mix_label = self.labels[rand_idx]

        mix_image = Image.open(mix_img_path).convert('RGB')
        mix_image = self.transform(mix_image)  # 应用与主图像相同的变换

        # 生成 MixUp 参数
        alpha = 0.4
        lam = np.random.beta(alpha, alpha)

        # 混合图像
        mixed_image = lam * image + (1 - lam) * mix_image

        # 混合标签
        mixed_label = lam * label + (1 - lam) * mix_label

        return mixed_image, mixed_label

    # 定义 CutMix 增强
    def cutmix(self, image, label):
        # 随机选择一个样本
        rand_idx = np.random.randint(0, len(self.image_paths))
        cut_img_path = self.image_paths[rand_idx]
        cut_label = self.labels[rand_idx]

        cut_image = Image.open(cut_img_path).convert('RGB')
        cut_image = self.transform(cut_image)  # 应用与主图像相同的变换

        # 生成 CutMix 参数
        alpha = 1.0
        lam = np.random.beta(alpha, alpha)

        # 获取图像尺寸
        _, w, h = image.size()
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        bw = int(w * np.sqrt(1 - lam))
        bh = int(h * np.sqrt(1 - lam))

        # 确定切割区域
        x1 = np.clip(cx - bw // 2, 0, w)
        y1 = np.clip(cy - bh // 2, 0, h)
        x2 = np.clip(cx + bw // 2, 0, w)
        y2 = np.clip(cy + bh // 2, 0, h)

        # 将切割区域替换为另一个图像的区域
        image_np = image.clone()
        cut_image_np = cut_image.clone()
        image_np[:, y1:y2, x1:x2] = cut_image_np[:, y1:y2, x1:x2]
        mixed_image = image_np

        # 调整标签
        mixed_label = lam * label + (1 - lam) * cut_label

        return mixed_image, mixed_label
