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

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
                        
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    # 使用tqdm创建进度条
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        
        # 更新进度条描述
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

def get_device(platform):
    if platform == 'cuda' and torch.cuda.is_available():
        return torch.device("cuda")
    elif platform == 'mps' and torch.backends.mps.is_available():
        return torch.device("mps")
    elif platform == 'directml':
        import torch_directml
        return torch_directml.device(torch_directml.default_device())
    else:
        return torch.device("cpu")

def main(args):
    # 设置超参数
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    
    device = get_device(args.platform)
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
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 训练循环
    start_time = time.time()
    for epoch in range(num_epochs):
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

    # 保存模型
    torch.save(model.state_dict(), 'vit_df_scanner.pth')
    print("Training completed. Model saved as 'vit_df_scanner.pth'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ViT-DF-Scanner model.")
    parser.add_argument('--platform', type=str, default='cpu', choices=['cpu', 'cuda', 'mps', 'directml'],
                        help='Computation platform to use (default: cpu)')
    args = parser.parse_args()
    main(args)