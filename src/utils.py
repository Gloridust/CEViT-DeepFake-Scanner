# utils.py

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast  # 修正导入路径
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score  # 添加 roc_auc_score

# 添加 Focal Loss 实现
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt 是预测正确的概率
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

def train(model, data_loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0

    for images, labels in tqdm(data_loader, desc='Training'):
        images = images.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
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
            outputs = model(images)  # [batch_size]
            preds = torch.sigmoid(outputs)  # [batch_size]
            predictions.extend(preds.cpu().numpy())

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
            labels = labels.to(device).float()

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(preds)
            all_preds.extend((preds > 0.5).astype(int))
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

def find_best_threshold(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    youden_index = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_index)]
    return best_threshold
