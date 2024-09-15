# models.py 定义了模型结构，包括ConvNext和RepLKNet的组合

import torch
from torch import nn
import timm

class AugmentInputsNetwork(nn.Module):
    def __init__(self, model):
        super(AugmentInputsNetwork, self).__init__()
        self.model = model
        self.adapter = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.adapter(x)
        x = (x - torch.tensor(timm.data.IMAGENET_DEFAULT_MEAN).view(1, -1, 1, 1).to(x.device)) / \
            torch.tensor(timm.data.IMAGENET_DEFAULT_STD).view(1, -1, 1, 1).to(x.device)
        return self.model(x)

class FinalModel(nn.Module):
    def __init__(self):
        super(FinalModel, self).__init__()

        self.convnext = timm.create_model('convnext_base', pretrained=True, num_classes=1)
        self.convnext = AugmentInputsNetwork(self.convnext)

        self.replknet = timm.create_model('replknet_31b', pretrained=True, num_classes=1)
        self.replknet = AugmentInputsNetwork(self.replknet)

    def forward(self, x):
        B, C, H, W = x.shape

        pred1 = self.convnext(x)
        pred2 = self.replknet(x)

        outputs = torch.stack((pred1, pred2), dim=-1).mean(dim=-1)
        return outputs.squeeze()
