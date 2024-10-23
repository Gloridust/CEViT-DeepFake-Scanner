# models.py

import torch
from torch import nn
import timm

class FinalModel(nn.Module):
    def __init__(self):
        super(FinalModel, self).__init__()
        
        # 使用更强大的基础模型
        self.convnext = timm.create_model('convnext_base', pretrained=True, num_classes=0)  # 设置num_classes=0移除最后的分类层
        self.efficientnet = timm.create_model('efficientnet_b2', pretrained=True, num_classes=0)
        self.vit = timm.create_model('vit_small_patch16_384', pretrained=True, num_classes=0)
        
        # 获取每个模型的特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 384, 384)
            conv_dim = self.convnext(dummy_input).shape[1]  # 通常是1024
            eff_dim = self.efficientnet(dummy_input).shape[1]  # 通常是1408
            vit_dim = self.vit(dummy_input).shape[1]  # 通常是384
        
        total_dim = conv_dim + eff_dim + vit_dim  # 2816
        
        # 特征融合层 - 渐进式降维
        self.fusion = nn.Sequential(
            # 第一层：保持较大维度以保留特征信息
            nn.Linear(total_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 第二层：中等维度的特征整合
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            # 第三层：进一步提取关键特征
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            # 最终分类层
            nn.Linear(256, 2)
        )
        
        # 初始时冻结基础模型参数
        self.freeze_base_models()
    
    def freeze_base_models(self):
        """冻结所有基础模型的参数"""
        for param in self.convnext.parameters():
            param.requires_grad = False
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False
    
    def unfreeze_base_models(self):
        """解冻所有基础模型的参数"""
        for param in self.convnext.parameters():
            param.requires_grad = True
        for param in self.efficientnet.parameters():
            param.requires_grad = True
        for param in self.vit.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        conv_features = self.convnext(x)
        eff_features = self.efficientnet(x)
        vit_features = self.vit(x)
        
        # 特征融合
        combined = torch.cat((conv_features, eff_features, vit_features), dim=1)
        output = self.fusion(combined)
        
        return output
