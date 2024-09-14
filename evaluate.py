# evaluate.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from vit_df_scanner import ViTDFScanner
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, f1, cm

def visualize_attention(model, image, device):
    model.eval()
    image = image.unsqueeze(0).to(device)
    
    # 获取注意力权重
    with torch.no_grad():
        attentions = model.vit_model.blocks[-1].attn.attn

    # 可视化注意力图
    att_mat = attentions[0, 0, 1:].reshape(14, 14)
    att_mat = att_mat.cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze().permute(1, 2, 0).cpu())
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(att_mat, cmap='viridis')
    plt.title("Attention Map")
    plt.axis('off')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

def main():
    # 设置超参数
    batch_size = 16
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # N卡
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Apple

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载测试数据集
    test_dataset = datasets.ImageFolder(root='./data/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 加载模型
    model = ViTDFScanner().to(device)
    model.load_state_dict(torch.load('vit_df_scanner.pth', map_location=device, weights_only=True))

    # 评估模型
    accuracy, f1, cm = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # 可视化注意力（这里假设使用测试集中的第一张图片）
    sample_image, _ = test_dataset[0]
    visualize_attention(model, sample_image, device)

if __name__ == "__main__":
    main()
