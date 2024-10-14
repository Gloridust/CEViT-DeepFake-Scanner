# CEViT-DeepFake-Scanner

CEViT-DeepFake-Scanner 是一个基于深度学习的工具，用于检测图像中的AI生成人脸。它结合使用ConvNeXt和EfficientNet模型，以实现高准确度地区分真实和AI生成的人脸图像。

## 目录

- [CEViT-DeepFake-Scanner](#CEViT-DeepFake-Scanner)
  - [目录](#目录)
  - [特性](#特性)
  - [安装](#安装)
  - [使用方法](#使用方法)
    - [训练](#训练)
    - [推理](#推理)
  - [项目结构](#项目结构)
  - [许可证](#许可证)

## 特性

- 利用最先进的深度学习模型（ConvNeXt和EfficientNet）
- 支持训练和推理模式
- 实现数据增强和混合精度训练以提高性能
- 在验证过程中提供详细的指标（准确率、精确率、召回率、F1分数）
- 性能指标：
  - Loss: 0.0173
  - Accuracy: 0.9941
  - Precision: 0.9985
  - Recall: 0.9942
  - F1 Score: 0.9964

## 安装

1. 克隆仓库：
   ```
   git clone https://github.com/yourusername/CEViT-DeepFake-Scanner.git
   cd CEViT-DeepFake-Scanner
   ```

2. 创建虚拟环境（可选但推荐）：
   ```
   python -m venv venv
   source venv/bin/activate  # 在Windows上，使用 `venv\Scripts\activate`
   ```

3. 安装所需的包：
   ```
   pip install -r requirements.txt
   ```

## 使用方法

### 训练

使用 `main_train.py` 脚本来训练模型：

```
python src/main_train.py --data_dir 训练数据路径 --batch_size 8 --epochs 20 --device cuda --lr 0.0005
```

参数：
- `--data_dir`：训练数据目录的路径
- `--batch_size`：训练的批量大小（默认：8）
- `--epochs`：训练的轮数（默认：20）
- `--device`：用于训练的设备（cuda、mps或cpu；默认：cuda）
- `--lr`：学习率（默认：0.0005）

### 推理

使用 `main.py` 脚本对一组图像进行推理：

```
python src/main.py --input_dir 输入图像路径 --output_csv 输出文件路径.csv --device cuda --model_path 模型检查点路径.pth
```

参数：
- `--input_dir`：包含输入图像的目录路径（默认：/testdata）
- `--output_csv`：保存输出CSV文件的路径（默认：./cla_pre.csv）
- `--device`：用于推理的设备（cuda、mps或cpu；默认：cuda）
- `--model_path`：训练好的模型检查点的路径（默认：./src/checkpoint_epoch_9.pth）

## 项目结构

```
CEViT-DeepFake-Scanner/
├── src/
│   ├── dataset.py      # 数据集加载和预处理
│   ├── main.py         # 推理脚本
│   ├── main_train.py   # 训练脚本
│   ├── models.py       # 模型架构定义
│   └── utils.py        # 用于训练和评估的实用函数
├── doc/
│   └── article.md      # 项目相关文档或文章
├── requirements.txt    # Python依赖
└── README.md           # 本文件
```

## 许可证

[MIT](LICENSE)
