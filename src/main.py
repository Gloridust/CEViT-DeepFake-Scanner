# main.py 用于执行推理并生成结果文件

import os
import pandas as pd
import argparse
import torch
from models import FinalModel
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from utils import find_best_threshold

def infer_and_save_results(model, input_dir, output_csv, device, threshold):
    # 定义测试时的数据增强变换列表
    tta_transforms = [
        transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Lambda(lambda img: img.rotate(15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Lambda(lambda img: img.rotate(-15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        # 可以根据需要添加更多变换
    ]

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    print(f"扫描到的图片数量: {len(image_files)}")
    results = []

    for filename in tqdm(image_files, desc='Processing images'):
        image_path = os.path.join(input_dir, filename)
        original_image = Image.open(image_path).convert('RGB')

        probs = []

        with torch.no_grad():
            for transform in tta_transforms:
                augmented_image = transform(original_image)
                augmented_image = augmented_image.unsqueeze(0).to(device)
                output = model(augmented_image)
                prob = torch.sigmoid(output).item()  # 保留 sigmoid 用于概率
                probs.append(prob)

            avg_prob = sum(probs) / len(probs)  # 计算平均概率

            result = 1 if avg_prob > threshold else 0  # 使用可调节的阈值
            results.append((os.path.splitext(filename)[0], result))  # 保存文件名（无扩展名）和结果

    # 保存结果到 CSV 文件
    results.sort()  # 按字典序排序
    df = pd.DataFrame(results, columns=['filename', 'result'])
    df.to_csv(output_csv, index=False, header=False)

def main():
    parser = argparse.ArgumentParser(description='AI-Generated Face Detection Inference')
    parser.add_argument('--input_dir', type=str, default='/testdata', help='Path to input directory containing images')
    parser.add_argument('--output_csv', type=str, default='./cla_pre.csv', help='Path to output CSV file')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'], help='Device to use for inference')
    parser.add_argument('--model_path', type=str, default='./src/checkpoints/best_model.pth', help='Path to the trained model')  # 修改默认模型路径
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for classification')  # 添加阈值参数

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU.")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS is not available. Switching to CPU.")
        args.device = 'cpu'

    device = torch.device(args.device)

    # 加载模型
    model = FinalModel().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    # 使用验证集找到最佳阈值
    # 假设有一个验证函数
    val_labels, val_probs = get_validation_data()  # 需要实现
    best_threshold = find_best_threshold(val_labels, val_probs)
    print(f"最佳阈值: {best_threshold:.4f}")

    # 执行推理并保存结果
    infer_and_save_results(model, args.input_dir, args.output_csv, device, threshold=best_threshold)

def get_validation_data():
    # 实现获取验证集标签和概率的函数
    # 这里假设您有一个函数或数据来源来获取验证集的标签和模型预测的概率
    # 例如，从验证集加载器中获取
    # 需要根据您的具体情况进行实现
    raise NotImplementedError("请实现 get_validation_data 函数以返回验证集的标签和预测概率。")

if __name__ == '__main__':
    main()
