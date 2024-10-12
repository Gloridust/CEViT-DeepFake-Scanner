# main_train.py 用于训练模型的主程序

import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models import FinalModel
from dataset import FaceDataset
from utils import train, validate
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        pred = pred.unsqueeze(-1)  # 确保 pred 是 2D: [batch_size, 1]
        target = target.unsqueeze(-1)  # 确保 target 是 2D: [batch_size, 1]
        
        # 创建 smooth labels
        smooth_target = torch.zeros_like(pred)
        smooth_target.fill_(self.smoothing)
        smooth_target.scatter_(1, target.long(), 1 - self.smoothing)
        
        # 计算 BCE 损失
        loss = F.binary_cross_entropy_with_logits(pred, smooth_target, reduction='mean')
        return loss

def main():
    parser = argparse.ArgumentParser(description='AI-Generated Face Detection Training')
    parser.add_argument('--data_dir', type=str, default='data/train', help='Path to training data')
    parser.add_argument('--val_data_dir', type=str, default='data/val', help='Path to validation data')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU.")
        args.device = 'cpu'

    device = torch.device(args.device)

    # 数据集和数据加载器
    train_dataset = FaceDataset(args.data_dir, train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_dataset = FaceDataset(args.val_data_dir, train=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 模型、损失函数和优化器
    model = FinalModel().to(device)
    criterion = LabelSmoothingLoss(smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # 使用 CosineAnnealingLR 调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = GradScaler(enabled=(args.device=='cuda'))

    # 开始训练
    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer, device, scheduler, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch}/{args.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

        # 保存最后一个 epoch 的模型
        torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pth')

if __name__ == '__main__':
    main()
