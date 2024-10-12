# utils.py

import torch
from tqdm import tqdm
from torch.cuda.amp import autocast

def train(model, data_loader, criterion, optimizer, device, scheduler, scaler):
    model.train()
    total_loss = 0

    for images, labels in tqdm(data_loader, desc='Training'):
        images = images.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()

        with autocast(enabled=scaler.is_enabled()):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def inference(model, data_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, _ in tqdm(data_loader, desc='Inference'):
            images = images.to(device)
            outputs = model(images)  # [batch_size]
            preds = torch.sigmoid(outputs)  # [batch_size]
            predictions.extend(preds.cpu().numpy())

    return predictions

def adjust_learning_rate(optimizer, epoch, initial_lr):
    """学习率调度函数，可根据需要进行调整"""
    lr = initial_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

@torch.no_grad()
def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(data_loader, desc='Validating'):
        images = images.to(device)
        labels = labels.to(device).float()

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        predicted = (outputs > 0).float()  # 使用 0 作为阈值进行预测
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy
