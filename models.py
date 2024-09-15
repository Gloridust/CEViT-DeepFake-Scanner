# models.py

import torch
from torch import nn
import timm

class AugmentInputsNetwork(nn.Module):
    def __init__(self, model):
        super(AugmentInputsNetwork, self).__init__()
        self.model = model
        self.adapter = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

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

        self.efficientnet = timm.create_model('efficientnet_b3', pretrained=True, num_classes=1)
        self.efficientnet = AugmentInputsNetwork(self.efficientnet)

    def forward(self, x):
        pred1 = self.convnext(x)  # [batch_size, 1]
        pred2 = self.efficientnet(x)  # [batch_size, 1]

        # Stack and average
        outputs = torch.stack((pred1, pred2), dim=-1).mean(dim=-1)  # [batch_size, 1]
        outputs = outputs.squeeze(1)  # [batch_size]

        # 确保输出为 float32
        assert outputs.dtype == torch.float32, f"Output dtype is {outputs.dtype}, expected torch.float32"

        return outputs
