# models.py

import torch
from torch import nn
import timm

class AugmentInputsNetwork(nn.Module):
    def __init__(self, model):
        super(AugmentInputsNetwork, self).__init__()
        self.model = model

    def forward(self, x):
        # 移除额外的卷积层，因为我们已经在数据集中添加了标准化
        return self.model(x)

class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class FinalModel(nn.Module):
    def __init__(self):
        super(FinalModel, self).__init__()
        
        self.backbone = timm.create_model('efficientnetv2_rw_s', pretrained=True, features_only=True)
        
        self.fpn_layers = nn.ModuleList([
            FPN(in_channels, 256) for in_channels in self.backbone.feature_info.channels()
        ])
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256 * len(self.fpn_layers), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        fpn_features = [fpn(feature) for fpn, feature in zip(self.fpn_layers, features)]
        combined = torch.cat([nn.AdaptiveAvgPool2d(1)(f) for f in fpn_features], dim=1)
        output = self.classifier(combined)
        return output.squeeze(-1)  # 返回 [batch_size] 形式的输出
