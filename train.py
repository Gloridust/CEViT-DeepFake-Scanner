# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from vit_df_scanner import ViTDFScanner
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import time
import argparse
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def custom_optimizer_step(optimizer, model):
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.sub_(param.grad * optimizer.param_groups[0]['lr'])

def train_one_epoch(model, dataloader, criterion, optimizer, device, accumulation_steps=4):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        
        # Normalize the loss to account for accumulation steps
        loss = loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            custom_optimizer_step(optimizer, model)
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(output.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        
        progress_bar.set_description(f"Training - Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, f1

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
   # 使用tqdm创建进度条
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)

    with torch.no_grad():
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
            # 更新进度条描述
            progress_bar.set_description(f"Validating - Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, f1

def get_device(platform, allow_virtual_memory=False):
    if platform == 'cuda' and torch.cuda.is_available():
        if allow_virtual_memory:
            torch.cuda.set_per_process_memory_fraction(1.0, 0)  # 允许使用所有可用内存
        return torch.device("cuda")
    elif platform == 'mps' and torch.backends.mps.is_available():
        return torch.device("mps")
    elif platform == 'directml':
        import torch_directml
        if allow_virtual_memory:
            os.environ['PYTORCH_DIRECTML_ALLOW_SYSTEM_FALLBACK'] = '1'
        return torch_directml.device(torch_directml.default_device())
    else:
        return torch.device("cpu")

def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, filename)

def load_checkpoint(model, optimizer, scheduler, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']

def main(args):
    # 设置超参数
    batch_size = 16  # 减小批量大小
    num_epochs = 50
    learning_rate = 1e-3  # 为 SGD 调整学习率
    
    device = get_device(args.platform, args.allow_virtual_memory)
    print(f"Using device: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载数据集
    train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
    val_dataset = datasets.ImageFolder(root='data/val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 初始化模型
    model = ViTDFScanner().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  # 使用 SGD 代替 AdamW

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 使用 StepLR 代替 CosineAnnealingLR

    # 如果指定了恢复训练，则加载检查点
    start_epoch = 0
    if args.resume:
        start_epoch, train_loss, val_loss = load_checkpoint(model, optimizer, scheduler, args.resume)
        print(f"Resuming from epoch {start_epoch+1}")

    # 训练循环
    start_time = time.time()
    last_save_time = start_time
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)
        
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s, Total Time: {total_time:.2f}s")
        print(f"Progress: {(epoch+1)/num_epochs*100:.2f}%")

        # 每10分钟保存一次检查点
        if time.time() - last_save_time > 600:  # 600 seconds = 10 minutes
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, f'checkpoint_epoch_{epoch+1}.pth')
            last_save_time = time.time()
            print(f"Checkpoint saved at epoch {epoch+1}")

        # 检查是否需要手动保存
        if args.save_now:
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, 'manual_save.pth')
            print("Manual save completed")
            args.save_now = False  # 重置标志

    # 保存最终模型
    torch.save(model.state_dict(), 'vit_df_scanner_final.pth')
    print("Training completed. Final model saved as 'vit_df_scanner_final.pth'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ViT-DF-Scanner model.")
    parser.add_argument('--platform', type=str, default='cpu', choices=['cpu', 'cuda', 'mps', 'directml'],
                        help='Computation platform to use (default: cpu)')
    parser.add_argument('--allow_virtual_memory', action='store_true',
                        help='Allow the use of virtual memory (system RAM) for GPU computations')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file for resuming training')
    parser.add_argument('--save_now', action='store_true',
                        help='Manually save a checkpoint during the next iteration')
    args = parser.parse_args()
    main(args)