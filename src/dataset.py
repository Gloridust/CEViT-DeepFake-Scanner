# dataset.py
# 定义了数据集的加载和预处理

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class FaceDataset(Dataset):
    def __init__(self, data_dir, train=True):
        self.data_dir = data_dir
        self.train = train
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
                self.labels.append(1)  # 真实人脸标签为1

        for img_name in os.listdir(fake_dir):
            if img_name.lower().endswith(valid_image_extensions):
                self.image_paths.append(os.path.join(fake_dir, img_name))
                self.labels.append(0)  # AI生成人脸标签为0

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),  # 调整为256以提高效率
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),  # 调整为256以提高效率
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # 确保标签为 float32 类型的张量
        label = torch.tensor(label, dtype=torch.float32)

        return image, label
