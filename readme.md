# ViT-DF-Scanner: Vision Transformer-based Deep Fake Detection System

## 项目概述

ViT-DF-Scanner 是一个基于 Vision Transformer (ViT) 的 DeepFake 检测系统。该项目旨在利用先进的计算机视觉技术来识别和分类深度伪造图像，为维护数字媒体的真实性和可信度提供有力工具。

## 科学原理

### 1. Vision Transformer (ViT)

ViT 是一种将 Transformer 架构应用于计算机视觉任务的创新模型。不同于传统的卷积神经网络 (CNN)，ViT 将输入图像分割成固定大小的块（patches），然后将这些块线性嵌入并添加位置编码。这种方法允许模型捕捉图像的全局上下文信息，这对于检测深度伪造特别有用。

主要组件包括：
- Patch Embedding：将图像分割成块并进行线性投影
- Position Embedding：为每个块添加位置信息
- Transformer Encoder：包含多头自注意力机制和前馈网络

### 2. 特殊图像处理技术

#### 2.1 亮度和对比度调整

通过调整图像的亮度和对比度，我们可以增强某些可能被深度伪造算法改变的细微特征。这种预处理步骤有助于模型更好地识别伪造痕迹。

#### 2.2 SRM (Spatial Rich Model) 滤波器

SRM 滤波器是一组用于提取图像残差特征的高通滤波器。这些滤波器能够有效地捕捉图像中的细微变化和不一致性，这些特征通常是深度伪造算法留下的痕迹。

### 3. 分类头部

分类头部使用全局平均池化和全连接层来处理 ViT 提取的特征，并输出最终的分类结果。

## 系统架构

ViT-DF-Scanner 系统由以下主要组件组成：

1. 图像预处理模块：包括亮度/对比度调整和 SRM 滤波
2. ViT 主干网络：用于特征提取
3. 分类头部：用于最终的真伪判断

## 操作步骤

### 环境设置

1. 克隆仓库：
   ```
   git clone https://github.com/Gloridust/ViT-DF-Scanner.git
   cd ViT-DF-Scanner
   ```

2. 创建并激活虚拟环境（可选但推荐）：
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

### 数据准备

1. 收集深度伪造图像和真实图像数据集。
2. 将数据集组织成以下结构：
   ```
   data/
   ├── train/
   │   ├── real/
   │   └── fake/
   ├── val/
   │   ├── real/
   │   └── fake/
   └── test/
       ├── real/
       └── fake/
   ```

3. 更新 `train.py` 和 `evaluate.py` 中的数据路径。

### 模型训练

1. 配置超参数：
   打开 `train.py`，根据需要调整 `batch_size`、`num_epochs`、`learning_rate` 等参数。

2. 开始训练：
   ```
   python train.py
   ```

3. 监控训练进度：
   脚本将输出每个 epoch 的训练损失、准确率和 F1 分数，以及验证集上的相应指标。

### 可选参数

1. 自动保存和手动保存：
   - 每10分钟自动保存一次检查点。
   - 通过命令行参数 `--save_now` 可以触发手动保存。
   - 保存的检查点包含模型状态、优化器状态、学习率调度器状态等，允许从任何保存点继续训练。

2. 使用虚拟显存：
   - 添加了 `--allow_virtual_memory` 参数，当使用时，允许 GPU 使用系统内存作为虚拟显存。
   - 对于 DirectML，设置了环境变量 `PYTORCH_DIRECTML_ALLOW_SYSTEM_FALLBACK` 来允许使用系统内存。

3. 恢复训练：
   - 添加了 `--resume` 参数，可以指定检查点文件来恢复训练。

使用方法示例：

1. 正常训练：
   ```
   python train.py --platform directml --allow_virtual_memory
   ```

2. 恢复训练：
   ```
   python train.py --platform directml --allow_virtual_memory --resume checkpoint_epoch_10.pth
   ```

3. 手动保存：
   ```
   python train.py --platform directml --allow_virtual_memory --save_now
   ```

### 模型评估

1. 运行评估脚本：
   ```
   python evaluate.py
   ```

2. 分析结果：
   脚本将输出测试集上的准确率和 F1 分数，并显示混淆矩阵和注意力可视化。

### 使用训练好的模型

1. 加载模型：
   ```python
   from vit_df_scanner import ViTDFScanner
   import torch

   model = ViTDFScanner()
   model.load_state_dict(torch.load('vit_df_scanner.pth'))
   model.eval()
   ```

2. 进行预测：
   ```python
   from torchvision import transforms
   from PIL import Image

   transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
   ])

   image = Image.open('path_to_your_image.jpg')
   input_tensor = transform(image).unsqueeze(0)

   with torch.no_grad():
       output = model(input_tensor)
       prediction = torch.argmax(output, dim=1).item()

   print("Prediction: ", "Fake" if prediction == 1 else "Real")
   ```

## 项目结构

```
ViT-DF-Scanner/
├── vit_backbone.py
├── image_processing.py
├── vit_df_scanner.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## 许可证

本项目采用 MIT 许可证。详情请见 [LICENSE](LICENSE) 文件。

## 联系方式

如有任何问题或建议，请通过以下方式联系我们：

- 项目维护者：[Your Name](mailto:your.email@example.com)
- 项目 GitHub 仓库：[https://github.com/Gloridust/ViT-DF-Scanner](https://github.com/Gloridust/ViT-DF-Scanner)

## 致谢

- 感谢所有为本项目做出贡献的开发者和研究人员
- 特别感谢 [ViT 论文](https://arxiv.org/abs/2010.11929) 的作者们，他们的工作为本项目提供了重要基础
