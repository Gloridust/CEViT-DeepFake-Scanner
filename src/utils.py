# utils.py

import torch
from tqdm import tqdm
from torch.cuda.amp import autocast  # 修正导入路径
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train(model, data_loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0

    for images, labels in tqdm(data_loader, desc='Training'):
        images = images.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

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

def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device).float()

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    precision = precision_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    recall = recall_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
    f1 = f1_score(all_labels, (np.array(all_preds) > 0.5).astype(int))

    print(f"Validation Results:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    model.train()
    return avg_loss
