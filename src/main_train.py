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

def main():
    parser = argparse.ArgumentParser(description='AI-Generated Face Detection Training')
    parser.add_argument('--data_dir', type=str, default='data/train', help='Path to training data')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'], help='Device to use for training')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')

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
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    # 数据集和数据加载器
    total_dataset = FaceDataset(args.data_dir, train=True)
    train_size = int(0.8 * len(total_dataset))
    val_size = len(total_dataset) - train_size
    train_dataset, val_dataset = random_split(total_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 模型、损失函数和优化器
    model = FinalModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 开始训练
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device, scaler)
        val_loss = validate(model, val_loader, criterion, device)  # 添加验证步骤
        scheduler.step()

        print(f"Epoch [{epoch}/{args.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 保存模型
        torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pth')

if __name__ == '__main__':
    main()
