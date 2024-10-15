# main_train.py 用于训练模型的主程序

import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from models import FinalModel
from dataset import FaceDataset
from utils import train, validate
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import numpy as np
import time  # 添加时间模块
import os  # 新增导入，用于处理文件路径
from sklearn.utils import class_weight

def main():
    parser = argparse.ArgumentParser(description='AI-Generated Face Detection Training')
    parser.add_argument('--data_dir', type=str, default='data/train', help='Path to training data')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')  # 增加 epoch 数量
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'], help='Device to use for training')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')  # 调整学习率
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')  # 新增参数

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

    # 打印训练和验证集的标签分布
    train_labels = [train_dataset.dataset.labels[i] for i in train_dataset.indices]
    val_labels = [val_dataset.dataset.labels[i] for i in val_dataset.indices]
    print(f"训练集标签分布: {np.bincount(train_labels)}")  # [真实人脸数量, AI生成人脸数量]
    print(f"验证集标签分布: {np.bincount(val_labels)}")  # [真实人脸数量, AI生成人脸数量]

    # 计算类别权重
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"类别权重: {class_weights}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 模型、损失函数和优化器
    model = FinalModel().to(device)
    # 使用加权的 BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1].unsqueeze(0))  # 为正类设置权重
    optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1)  # 调低学习率

    # 冻结子模型的权重
    model.freeze_submodels()

    # 定义学习率调度器为 ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # 定义早停参数
    best_val_loss = np.inf
    patience = 5  # 提高耐心值以允许更多轮训练
    trigger_times = 0

    # 新增：加载检查点以继续训练
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"加载检查点 '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # 设置 strict=False
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            trigger_times = checkpoint['trigger_times']
            print(f"检查点 '{args.resume}' (epoch {checkpoint['epoch']}) 已加载，继续训练从第 {start_epoch} 轮开始。")
        else:
            print(f"未找到检查点 '{args.resume}'，从头开始训练。")
            start_epoch = 1
    else:
        start_epoch = 1  # 默认从一轮开始

    # 开始训练
    for epoch in range(start_epoch, args.epochs + 1):
        # 如果达到一定的epoch数，解冻子模型
        if epoch == 10:
            print("解冻子模型的权重，进行联合训练。")
            model.unfreeze_submodels()
            # 重新定义优化器以包含新的可训练参数
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr * 0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

        start_time = time.time()  # 记录epoch开始时间

        train_loss = train(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, accuracy, precision, recall, f1 = validate(model, val_loader, criterion, device)  # 解包新增的返回值

        # 更新学习率
        scheduler.step(val_loss)

        epoch_duration = time.time() - start_time  # 计算epoch时长

        print(f"Epoch [{epoch}/{args.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, "
              f"Duration: {epoch_duration:.2f}s")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'trigger_times': trigger_times
            }, 'best_model.pth')  # 修改保存方式，保存更多状态信息
            print("保存当前最佳模型")
        else:
            trigger_times += 1
            print(f"早停计数器：{trigger_times}")
            if trigger_times >= patience:
                print("触发早停，停止训练")
                break

        # 记录每个 epoch 的验证信息
        with open('training_log.txt', 'a') as log_file:
            log_file.write(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                           f"Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, "
                           f"F1 Score={f1:.4f}, Duration={epoch_duration:.2f}s\n")

        # 保存一个检查点，以便之后恢复
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'trigger_times': trigger_times
        }, f'checkpoint_epoch_{epoch}.pth')  # 保存当前轮次的检查点

if __name__ == '__main__':
    main()