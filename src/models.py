# models.py

import torch
from torch import nn
import timm

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if x.dim() == 4:
            batch, channels, _, _ = x.size()
            y = self.squeeze(x).view(batch, channels)
            y = self.excitation(y).view(batch, channels, 1, 1)
            return x * y.expand_as(x)
        elif x.dim() == 2:
            # 如果输入已经是2D的特征向量，不应用 SE 模块
            return x
        else:
            raise ValueError("Unsupported tensor dimensions for SEBlock")

class FinalModel(nn.Module):
    def __init__(self):
        super(FinalModel, self).__init__()

        # 移除分类头，仅输出特征
        self.convnext = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
        self.convnext_se = SEBlock(channels=self.convnext.num_features)  # 添加 SE 模块

        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.efficientnet_se = SEBlock(channels=self.efficientnet.num_features)  # 添加 SE 模块

        self.vit = timm.create_model('vit_small_patch16_384', pretrained=True, num_classes=0)
        # ViT 输出为二维特征向量，不应用 SEBlock

        # 添加 Dropout 层
        self.dropout = nn.Dropout(p=0.5)

        # 确认每个子模型的特征维度
        conv_features = self.convnext.num_features  # 通常为 768
        eff_features = self.efficientnet.num_features  # 通常为 1280
        vit_features = self.vit.num_features  # 通常为 768

        combined_feature_size = conv_features + eff_features + vit_features  # 768 + 1280 + 768 = 2816

        # 添加更复杂的融合层
        self.fusion = nn.Sequential(
            nn.Linear(combined_feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # 提取 ConvNeXt 特征
        features1 = self.convnext.forward_features(x)  # [batch, convnext.num_features, H, W]
        features1 = self.convnext_se(features1)        # 应用 SE 模块
        features1 = nn.functional.adaptive_avg_pool2d(features1, (1,1)).view(x.size(0), -1)  # [batch, convnext.num_features]

        # 提取 EfficientNet 特征
        features2 = self.efficientnet.forward_features(x)  # [batch, efficientnet.num_features, H, W]
        features2 = self.efficientnet_se(features2)        # 应用 SE 模块
        features2 = nn.functional.adaptive_avg_pool2d(features2, (1,1)).view(x.size(0), -1)  # [batch, efficientnet.num_features]

        # 提取 ViT 特征
        features3 = self.vit.forward_features(x)  # 修改为 forward_features 以获取特征向量
        features3 = torch.flatten(features3, 1)   # 确保 ViT 输出为 [batch, vit.num_features]

        # 结合所有特征
        combined_features = torch.cat((features1, features2, features3), dim=1)  # [batch, combined_feature_size]

        # 应用融合层
        output = self.fusion(combined_features)
        return output.squeeze(1)  # [batch]
