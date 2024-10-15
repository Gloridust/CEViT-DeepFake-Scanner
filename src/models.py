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

class FinalModel(nn.Module):
    def __init__(self):
        super(FinalModel, self).__init__()

        # 使用更小的模型以减少显存占用
        self.convnext = timm.create_model('convnext_tiny', pretrained=True, num_classes=1)
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=1)
        self.vit = timm.create_model('vit_tiny_patch16_384', pretrained=True, num_classes=1)  # 替换为更小的 ViT 版本

        # 添加 Dropout 层
        self.dropout = nn.Dropout(p=0.5)

        # 添加可学习的权重参数
        self.weight_convnext = nn.Parameter(torch.ones(1))
        self.weight_efficientnet = nn.Parameter(torch.ones(1))

        # 修改组合层，包含 ViT 的输出
        self.combine = nn.Sequential(
            nn.Linear(3, 1),  # 从3个模型输出中组合
            nn.Sigmoid()
        )

    def forward(self, x):
        pred1 = self.convnext(x)
        pred2 = self.efficientnet(x)
        pred3 = self.vit(x)  # ViT 模型输出

        # 应用可学习的权重
        weighted_pred1 = self.weight_convnext * pred1
        weighted_pred2 = self.weight_efficientnet * pred2
        weighted_pred3 = self.weight_vit * pred3

        # 组合三个模型的加权输出
        combined = torch.cat((weighted_pred1, weighted_pred2, weighted_pred3), dim=1)

        # 应用 Dropout
        combined = self.dropout(combined)

        output = self.combine(combined)
        return output.squeeze(1)
