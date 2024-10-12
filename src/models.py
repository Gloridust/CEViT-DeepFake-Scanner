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

        self.convnext = timm.create_model('convnext_base', pretrained=True, num_classes=0)
        self.efficientnet = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)

        self.classifier = nn.Sequential(
            nn.Linear(self.convnext.num_features + self.efficientnet.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        feat1 = self.convnext(x)
        feat2 = self.efficientnet(x)
        combined = torch.cat((feat1, feat2), dim=1)
        return self.classifier(combined)
