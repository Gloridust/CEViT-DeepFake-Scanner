import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class ImageProcessingModule(nn.Module):
    """特殊图像处理模块"""
    def __init__(self, num_srm_filters=3):
        super().__init__()
        self.srm_filters = self.get_srm_filters(num_srm_filters)
        self.srm_conv = nn.Conv2d(1, num_srm_filters, kernel_size=5, padding=2, bias=False)
        self.srm_conv.weight.data = self.srm_filters
        self.srm_conv.weight.requires_grad = False

    def get_srm_filters(self, num_filters):
        """生成SRM滤波器"""
        srm_kernels = [
            # 第一个滤波器: 边缘检测
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 1, -4, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0]],
            # 第二个滤波器: 锐化
            [[-1, -1, -1, -1, -1],
             [-1, 2, 2, 2, -1],
             [-1, 2, 8, 2, -1],
             [-1, 2, 2, 2, -1],
             [-1, -1, -1, -1, -1]],
            # 第三个滤波器: 高斯差分
            [[0, 0, -1, 0, 0],
             [0, -1, -2, -1, 0],
             [-1, -2, 16, -2, -1],
             [0, -1, -2, -1, 0],
             [0, 0, -1, 0, 0]]
        ]
        srm_weights = torch.FloatTensor(srm_kernels[:num_filters]).unsqueeze(1)
        return srm_weights

    def adjust_brightness_contrast(self, x, brightness=1.0, contrast=1.0):
        """调整图像的亮度和对比度"""
        return TF.adjust_contrast(TF.adjust_brightness(x, brightness), contrast)

    def forward(self, x):
        """
        参数:
            x: 输入图像, 形状为 (batch_size, 3, height, width)
        返回:
            处理后的图像, 形状为 (batch_size, 3 + num_srm_filters, height, width)
        """
        # 调整亮度和对比度
        x_adjusted = self.adjust_brightness_contrast(x, brightness=1.1, contrast=1.1)
        
        # 应用SRM滤波器
        x_gray = TF.rgb_to_grayscale(x_adjusted)
        x_srm = self.srm_conv(x_gray)
        
        # 连接原始调整后的图像和SRM特征
        return torch.cat([x_adjusted, x_srm], dim=1)

class ClassificationHead(nn.Module):
    """分类头部"""
    def __init__(self, in_features, hidden_features, num_classes=2, dropout=0.1):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, num_classes)

    def forward(self, x):
        """
        参数:
            x: 输入特征, 形状为 (batch_size, in_features, height, width)
        返回:
            分类结果, 形状为 (batch_size, num_classes)
        """
        x = self.global_pool(x).flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
