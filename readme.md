# ViT-DF-Scanner: Vision Transformer-based Deep Fake Detection System

本项目旨在鉴别AI生成的人脸，使用了`ConvNext`和`EfficientNet`模型的组合。

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
python main_train.py --data_dir data/train --device cuda --batch_size 8 --epochs 30
```

- `--data_dir`：训练数据的路径。
- `--device`：使用的设备（`cuda`、`mps`或`cpu`）。
- `--batch_size`：批次大小。根据您的设备内存调整。对于 16G 显存的设备，批次大小可设为6～7，否则有概率爆显存。
- `--epochs`：训练轮数。

训练完成后，模型会以`checkpoint_epoch_{epoch}.pth`的形式保存在当前目录。

## 推理（测试）

运行以下命令进行推理：

```bash
python main_infer.py --data_dir data/test --device cuda --batch_size 8 --model_path checkpoint_epoch_30.pth
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

## 测试结果

以 512*512 规格的 3.12G 图片做训练，30 Epoch 后结果如下：

```
Test Results:
Accuracy: 0.8102
Precision: 0.9940
Recall: 0.7349
F1 Score: 0.8451
```

在此次模型测试中，该模型表现出诸多优秀之处。其准确率达到了 0.8102，表明模型在整体判断上具有较高的正确性。精度高达 0.9940，说明模型在预测为正例的样本中，真正的正例比例非常高。召回率为 0.7349，意味着模型能够成功识别出大部分的正例样本。F1 分数为 0.8451，综合考虑了精度和召回率，进一步体现了模型在准确性和全面性之间的良好平衡。总体而言，该模型在本次测试中展现出了较高的准确性、精度以及在识别正例样本方面的出色能力，同时在精度和召回率的平衡上也有较好的表现。