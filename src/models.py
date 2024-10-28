# models.py

import torch
from torch import nn
import timm

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x1, x2):
        # x1 作为 query，x2 作为 key 和 value
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        out = torch.matmul(attn, v)
        return out

class FeatureAttentionFusion(nn.Module):
    def __init__(self, conv_dim, eff_dim, vit_dim):
        super().__init__()
        
        # 将所有特征映射到相同的维度
        self.unified_dim = 512
        self.conv_proj = nn.Linear(conv_dim, self.unified_dim)
        self.eff_proj = nn.Linear(eff_dim, self.unified_dim)
        self.vit_proj = nn.Linear(vit_dim, self.unified_dim)
        
        # 交叉注意力模块
        self.conv_eff_attn = CrossAttention(self.unified_dim)
        self.conv_vit_attn = CrossAttention(self.unified_dim)
        self.eff_vit_attn = CrossAttention(self.unified_dim)
        
        # 特征融合后的处理
        self.fusion = nn.Sequential(
            nn.Linear(self.unified_dim * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
    def forward(self, conv_feat, eff_feat, vit_feat):
        # 投影到统一维度
        conv_proj = self.conv_proj(conv_feat)
        eff_proj = self.eff_proj(eff_feat)
        vit_proj = self.vit_proj(vit_feat)
        
        # 交叉注意力
        conv_eff = self.conv_eff_attn(conv_proj, eff_proj)
        conv_vit = self.conv_vit_attn(conv_proj, vit_proj)
        eff_vit = self.eff_vit_attn(eff_proj, vit_proj)
        
        # 特征融合
        fused_features = torch.cat([conv_eff, conv_vit, eff_vit], dim=1)
        return self.fusion(fused_features)

class FinalModel(nn.Module):
    def __init__(self):
        super(FinalModel, self).__init__()
        
        # 基础模型
        self.convnext = timm.create_model('convnext_base', pretrained=True, num_classes=0)
        self.efficientnet = timm.create_model('efficientnet_b2', pretrained=True, num_classes=0)
        self.vit = timm.create_model('vit_small_patch16_384', pretrained=True, num_classes=0)
        
        # 获取特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 384, 384)
            conv_dim = self.convnext(dummy_input).shape[1]
            eff_dim = self.efficientnet(dummy_input).shape[1]
            vit_dim = self.vit(dummy_input).shape[1]
        
        # 注意力融合层
        self.feature_fusion = FeatureAttentionFusion(conv_dim, eff_dim, vit_dim)
        
    def freeze_base_models(self):
        for param in self.convnext.parameters():
            param.requires_grad = False
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False
    
    def unfreeze_base_models(self):
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
        
        # 使用注意力机制融合特征
        output = self.feature_fusion(conv_features, eff_features, vit_features)
        return output
