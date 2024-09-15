# dataset.py 定义了数据集的加载和预处理

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FaceDataset(Dataset):
    def __init__(self, data_dir, train=True):
        self.data_dir = data_dir
        self.train = train
        self.image_paths = []
        self.labels = []

        # 假设真实人脸保存在'real'文件夹，AI生成的人脸保存在'fake'文件夹
        real_dir = os.path.join(data_dir, 'real')
        fake_dir = os.path.join(data_dir, 'fake')

        for img_name in os.listdir(real_dir):
            self.image_paths.append(os.path.join(real_dir, img_name))
            self.labels.append(1)  # 真实人脸标签为1

        for img_name in os.listdir(fake_dir):
            self.image_paths.append(os.path.join(fake_dir, img_name))
            self.labels.append(0)  # AI生成人脸标签为0

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = float(self.labels[idx])

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return image, label
