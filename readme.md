# ViT-DF-Scanner: Vision Transformer-based Deep Fake Detection System

本项目旨在鉴别AI生成的人脸，使用了`ConvNext`和`RepLKNet`模型的组合。

## 环境准备

1. **克隆或下载项目代码。**

2. **创建虚拟环境（可选但推荐）。**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **安装所需的Python库。**

   ```bash
   pip install -r requirements.txt
   ```

   请确保您的PyTorch版本支持CUDA和MPS。如果没有，请根据您的设备重新安装适合的PyTorch版本。

   - **CUDA（NVIDIA V100-16G）**

     ```bash
     pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
     ```

   - **MPS（Apple M3 16G Mac）**

     ```bash
     pip install torch torchvision torchaudio
     ```

## 数据准备

请按照以下目录结构准备您的数据：

```
data/
├── train/
│   ├── real/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── fake/
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
└── test/
    ├── real/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    └── fake/
        ├── img1.jpg
        ├── img2.jpg
        └── ...
```

- 将真实人脸图像放入`real`文件夹。
- 将AI生成的人脸图像放入`fake`文件夹。

## 训练模型

运行以下命令开始训练：

```bash
python main_train.py --data_dir data/train --device cuda --batch_size 16 --epochs 30
```

- `--data_dir`：训练数据的路径。
- `--device`：使用的设备（`cuda`、`mps`或`cpu`）。
- `--batch_size`：批次大小。根据您的设备内存调整。对于V100-16G，批次大小可设为32或64。对于M3 16G Mac，可设为8或16。
- `--epochs`：训练轮数。

训练完成后，模型会以`checkpoint_epoch_{epoch}.pth`的形式保存在当前目录。

## 推理（测试）

运行以下命令进行推理：

```bash
python main_infer.py --data_dir data/test --device cuda --batch_size 16 --model_path checkpoint_epoch_30.pth
```

- `--data_dir`：测试数据的路径。
- `--device`：使用的设备。
- `--batch_size`：批次大小。
- `--model_path`：训练好的模型路径。

推理结果将在控制台输出，您也可以修改`main_infer.py`以保存或处理预测结果。

## 注意事项

- **参数调整**：请根据您的设备性能调整批次大小和学习率。
- **设备选择**：如果您指定的设备不可用，程序会自动切换到CPU。
- **数据格式**：确保您的图像数据为RGB格式，大小为512x512。如果不是，请在`dataset.py`中修改预处理代码。

## 可能的改进

- **添加更多数据增强**：如随机裁剪、颜色抖动等。
- **优化模型结构**：尝试使用其他预训练模型或自定义模型。
- **调整超参数**：根据验证集性能调整学习率、优化器等超参数。

## 联系方式

如果您有任何问题或建议，请联系：

- Email: your_email@example.com

```

---

## 说明

- **多GPU支持**：当前代码支持单GPU训练，如果需要多GPU训练，可使用`torch.nn.DataParallel`或`DistributedDataParallel`。
- **模型选择**：模型使用了`timm`库中的预训练模型，您可以根据需要更换其他模型。
- **损失函数**：使用了`BCEWithLogitsLoss`，因为最终输出为单个值，表示类别的概率。
