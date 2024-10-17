# models.py

import torch
from torch import nn
import timm

class FinalModel(nn.Module):
    def __init__(self):
        super(FinalModel, self).__init__()

        # 使用更小的模型以减少显存占用
        self.convnext = timm.create_model('convnext_tiny', pretrained=True, num_classes=1)
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=1)
        self.vit = timm.create_model('vit_small_patch16_384', pretrained=True, num_classes=1)

        # 添加 Dropout 层
        self.dropout = nn.Dropout(p=0.5)

        # 修改组合层以适应 CrossEntropyLoss
        self.combine = nn.Linear(3, 2)  # 从3个模型输出中组合，输出2个类别

    def forward(self, x):
        pred1 = self.convnext(x)
        pred2 = self.efficientnet(x)
        pred3 = self.vit(x)  # ViT 模型输出

        # 组合三个模型的输出
        combined = torch.cat((pred1, pred2, pred3), dim=1)

        # 应用 Dropout
        combined = self.dropout(combined)

        output = self.combine(combined)
        return output  # 移除 squeeze 操作
