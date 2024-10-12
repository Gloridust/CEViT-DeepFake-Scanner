# main_train.py 用于训练模型的主程序

import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models import FinalModel
from dataset import FaceDataset
from utils import train
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

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

    # 数据集和数据加载器
    train_dataset = FaceDataset(args.data_dir, train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 模型、损失函数和优化器
    model = FinalModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 修正 GradScaler 的初始化
    scaler = GradScaler()

    # 开始训练
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device, scaler)
        scheduler.step()

        print(f"Epoch [{epoch}/{args.epochs}], Loss: {train_loss:.4f}")

        # 保存模型
        torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pth')

if __name__ == '__main__':
    main()
