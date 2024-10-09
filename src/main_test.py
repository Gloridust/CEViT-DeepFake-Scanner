# main_test.py 用于测试模型性能

import argparse
import torch
from torch.utils.data import DataLoader
from models import FinalModel
from dataset import FaceDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm

def test(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Testing'):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)

def main():
    parser = argparse.ArgumentParser(description='AI-Generated Face Detection Testing')
    parser.add_argument('--data_dir', type=str, default='data/test', help='Path to test data')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'], help='Device to use for testing')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU.")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS is not available. Switching to CPU.")
        args.device = 'cpu'

    device = torch.device(args.device)

    # 数据集和数据加载器
    test_dataset = FaceDataset(args.data_dir, train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 加载模型
    model = FinalModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # 开始测试
    predictions, labels = test(model, test_loader, device)

    # 计算性能指标
    predictions_binary = (predictions > 0.5).astype(int)
    accuracy = accuracy_score(labels, predictions_binary)
    precision = precision_score(labels, predictions_binary)
    recall = recall_score(labels, predictions_binary)
    f1 = f1_score(labels, predictions_binary)

    print(f"Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == '__main__':
    main()