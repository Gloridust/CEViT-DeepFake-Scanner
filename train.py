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
import signal

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 全局变量用于触发手动保存
save_flag = False

def signal_handler(signum, frame):
    global save_flag
    save_flag = True
    print("\nManual save triggered. Saving now...")

signal.signal(signal.SIGINT, signal_handler)

def safe_to_numpy(tensor):
    if hasattr(tensor, 'cpu'):
        tensor = tensor.cpu()
    if hasattr(tensor, 'detach'):
        tensor = tensor.detach()
    try:
        return np.array(tensor)
    except TypeError:
        return np.array(tensor.tolist())

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
        
        loss = loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            custom_optimizer_step(optimizer, model)
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(output.data, 1)
        all_preds.extend(safe_to_numpy(predicted))
        all_labels.extend(safe_to_numpy(target))
        
        progress_bar.set_description(f"Training - Loss: {loss.item():.4f}")

        # 检查是否需要手动保存
        if save_flag:
            print("\nSaving checkpoint due to manual interruption...")
            return total_loss, all_preds, all_labels  # 提前返回以触发保存

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, f1

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)

    with torch.no_grad():
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            all_preds.extend(safe_to_numpy(predicted))
            all_labels.extend(safe_to_numpy(target))
            
            progress_bar.set_description(f"Validating - Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, f1

def get_device(platform, allow_virtual_memory=False):
    if platform == 'cuda' and torch.cuda.is_available():
        if allow_virtual_memory:
            torch.cuda.set_per_process_memory_fraction(1.0, 0)
        return torch.device("cuda")
    elif platform == 'mps' and torch.backends.mps.is_available():
        return torch.device("mps")
    elif platform == 'directml':
        import torch_directml
        if allow_virtual_memory:
            os.environ['PYTORCH_DIRECTML_ALLOW_SYSTEM_FALLBACK'] = '1'
        device = torch_directml.device(torch_directml.default_device())
        torch.set_default_tensor_type(torch.FloatTensor)
        return device
    else:
        return torch.device("cpu")

def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, filename):
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, filename)
        print(f"Checkpoint saved successfully: {filename}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def load_checkpoint(model, optimizer, scheduler, filename):
    try:
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0, 0, 0

def main(args):
    global save_flag
    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-4
    accumulation_steps = 4
    save_interval = 600  # 每10分钟保存一次
    
    device = get_device(args.platform, args.allow_virtual_memory)
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
    val_dataset = datasets.ImageFolder(root='data/val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = ViTDFScanner().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    start_epoch = 0
    if args.resume:
        start_epoch, train_loss, val_loss = load_checkpoint(model, optimizer, scheduler, args.resume)
        print(f"Resuming from epoch {start_epoch+1}")

    start_time = time.time()
    last_save_time = start_time
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)
        
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps)
        if save_flag:
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, None, f'manual_save_epoch_{epoch+1}.pth')
            break  # 退出训练循环
        
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s, Total Time: {total_time:.2f}s")
        print(f"Progress: {(epoch+1)/num_epochs*100:.2f}%")

        # 自动保存逻辑
        if time.time() - last_save_time > save_interval:
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, f'checkpoint_epoch_{epoch+1}.pth')
            last_save_time = time.time()

    # 保存最终模型
    try:
        torch.save(model.state_dict(), 'vit_df_scanner.pth')
        print("Training completed. Final model saved as 'vit_df_scanner.pth'")
    except Exception as e:
        print(f"Error saving final model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ViT-DF-Scanner model.")
    parser.add_argument('--platform', type=str, default='cpu', choices=['cpu', 'cuda', 'mps', 'directml'],
                        help='Computation platform to use (default: cpu)')
    parser.add_argument('--allow_virtual_memory', action='store_true',
                        help='Allow the use of virtual memory (system RAM) for GPU computations')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file for resuming training')
    args = parser.parse_args()
    main(args)
