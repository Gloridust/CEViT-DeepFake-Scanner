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

        # 添加一个简单的加权组合层
        self.combine = nn.Linear(2, 1)

    def forward(self, x):
        pred1 = self.convnext(x)
        pred2 = self.efficientnet(x)

        # 组合两个模型的输出
        combined = torch.cat((pred1, pred2), dim=1)
        output = self.combine(combined)
        return output.squeeze(1)
