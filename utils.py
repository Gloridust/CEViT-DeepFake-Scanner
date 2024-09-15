# utils.py 训练和推理的辅助函数

import torch
from tqdm import tqdm

def train(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for images, labels in tqdm(data_loader, desc='Training'):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
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
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            predictions.extend(preds.cpu().numpy())

    return predictions

def adjust_learning_rate(optimizer, epoch, initial_lr):
    """学习率调度函数，可根据需要进行调整"""
    lr = initial_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
