# main_train.py 用于训练模型的主程序

import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler  # 添加 WeightedRandomSampler 导入
from models import FinalModel
from dataset import FaceDataset
from utils import train, validate, FocalLoss  # 更新导入，包含 FocalLoss
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import random
import numpy as np
import time  # 添加时间模块
import os  # 新增导入，用于处理文件路径

def main():
    parser = argparse.ArgumentParser(description='AI-Generated Face Detection Training')
    parser.add_argument('--data_dir', type=str, default='data/train', help='Path to training data')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')  # 增加 epoch 数量
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'], help='Device to use for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')  # 调整学习率
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')  # 新增参数
    parser.add_argument('--use_sampler', action='store_true', help='Use WeightedRandomSampler for handling class imbalance')  # 添加是否使用采样器的参数
    parser.add_argument('--use_focal_loss', action='store_true', help='Use Focal Loss instead of BCEWithLogitsLoss')  # 添加使用 Focal Loss 的参数

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU.")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS is not available. Switching to CPU.")
        args.device = 'cpu'

    device = torch.device(args.device)

    # 根据设备类型决定是否使用混合精度
    if device.type == 'cuda':
        scaler = GradScaler()
    else:
        scaler = None

    # 数据集和数据加载器
    total_dataset = FaceDataset(args.data_dir, train=True)
    train_size = int(0.8 * len(total_dataset))
    val_size = len(total_dataset) - train_size
    train_dataset, val_dataset = random_split(total_dataset, [train_size, val_size])

    # 计算训练集中的正负样本数量
    train_labels = [total_dataset.labels[i] for i in train_dataset.indices]
    val_labels = [total_dataset.labels[i] for i in val_dataset.indices]

    num_train_positive = sum(train_labels)
    num_train_negative = len(train_labels) - num_train_positive
    num_val_positive = sum(val_labels)
    num_val_negative = len(val_labels) - num_val_positive

    # 打印训练和验证集的样本数量
    print(f"训练集样本数量: 正样本={num_train_positive}, 负样本={num_train_negative}")
    print(f"验证集样本数量: 正样本={num_val_positive}, 负样本={num_val_negative}")

    # 计算类别权重
    if num_train_positive > 0:
        pos_weight = torch.tensor([num_train_negative / num_train_positive]).to(device)
    else:
        pos_weight = None  # 或设置为一个默认值，例如 torch.tensor([1.0]).to(device)

    if args.use_sampler:
        class_counts = np.bincount(train_labels)
        class_weights = 1. / class_counts
        samples_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=8)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # 模型、损失函数和优化器
    model = FinalModel().to(device)
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')  # 使用 Focal Loss
    else:
        if pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 使用带权重的 BCE Loss
        else:
            criterion = nn.BCEWithLogitsLoss()  # 不使用权重

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 定义学习率调度器为 ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)

    # 定义早停参数
    best_val_loss = np.inf
    best_val_auc = 0.0  # 添加最佳 ROC-AUC
    patience = 4
    trigger_times = 0

    # 新增：加载检查点以继续训练
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"加载检查点 '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            best_val_auc = checkpoint.get('best_val_auc', 0.0)  # 加载最佳 ROC-AUC
            trigger_times = checkpoint['trigger_times']
            print(f"检查点 '{args.resume}' (epoch {checkpoint['epoch']}) 已加载，继续训练从第 {start_epoch} 轮开始。")
        else:
            print(f"未找到检查点 '{args.resume}'，从头开始训练。")
            start_epoch = 1
    else:
        start_epoch = 1  # 默认从第一轮开始

    # 开始训练
    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()  # 记录epoch开始时间

        train_loss = train(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, accuracy, precision, recall, f1, val_auc = validate(model, val_loader, criterion, device)  # 修改 validate 以返回 val_auc

        # 更新学习率
        scheduler.step(val_loss)

        epoch_duration = time.time() - start_time  # 计算epoch时长

        print(f"Epoch [{epoch}/{args.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
              f"F1 Score: {f1:.4f}, ROC-AUC: {val_auc:.4f}, Duration: {epoch_duration:.2f}s")

        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            trigger_times = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_auc': best_val_auc,
                'trigger_times': trigger_times
            }, './src/checkpoints/best_model.pth')
            print("保存当前最佳模型 (基于 ROC-AUC)")
        else:
            trigger_times += 1
            print(f"早停计数器：{trigger_times}")
            if trigger_times >= patience:
                print("触发早停，停止训练")
                break

        # 记录每个 epoch 的验证信息
        with open('./src/training_log.txt', 'a') as log_file:
            log_file.write(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                           f"Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, "
                           f"F1 Score={f1:.4f}, ROC-AUC={val_auc:.4f}, Duration={epoch_duration:.2f}s\n")

        # 保存一个检查点，以便之后恢复
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_auc': best_val_auc,
            'trigger_times': trigger_times
        }, f'./src/checkpoints/checkpoint_epoch_{epoch}.pth')

if __name__ == '__main__':
    main()
