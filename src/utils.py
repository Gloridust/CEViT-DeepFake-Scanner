# utils.py

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast  # 修正导入路径
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve  # 添加 roc_auc_score 和 roc_curve

# 移除 FocalLoss 实现

def train(model, data_loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0

    for images, labels in tqdm(data_loader, desc='Training'):
        images = images.to(device)
        labels = labels.to(device)  # 保持标签为 long 类型

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # 解缩梯度以进行裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 添加梯度裁剪
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 添加梯度裁剪
            optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

def inference(model, data_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, _ in tqdm(data_loader, desc='Inference'):
            images = images.to(device)
            outputs = model(images)  # [batch_size, 2]
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 获取第二类的概率
            predictions.extend(probs.cpu().numpy())

    return predictions

def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []  # 添加保存概率的列表

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 获取第二类的概率
            preds = np.argmax(outputs.cpu().numpy(), axis=1)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)  # 计算 ROC-AUC

    print(f"Validation Results:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")  # 显示 ROC-AUC

    model.train()
    return avg_loss, accuracy, precision, recall, f1, roc_auc  # 修改返回值，包含 ROC-AUC
