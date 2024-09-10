# vit_df_scanner.py

import torch
import torch.nn as nn
from vit_backbone import ViTBackbone
from image_processing import ImageProcessingModule
from image_processing import ClassificationHead

class ViTDFScanner(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=2, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, num_srm_filters=3, hidden_features=256):
        super().__init__()
        self.img_proc_model = ImageProcessingModule(num_srm_filters=num_srm_filters)
        self.vit_model = ViTBackbone(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans + num_srm_filters,
            num_classes=0, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias
        )
        self.classification_head = ClassificationHead(
            in_features=embed_dim, hidden_features=hidden_features, num_classes=num_classes
        )

    def forward(self, x):
        x_processed = self.img_proc_model(x)
        features = self.vit_model(x_processed)
        output = self.classification_head(features)
        return output
