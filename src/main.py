# main.py 用于执行推理并生成结果文件

import os
import pandas as pd
import argparse
import torch
from models import FinalModel
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def infer_and_save_results(model, input_dir, output_csv, device):
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
                prob = torch.sigmoid(output).item()
                probs.append(prob)

            avg_prob = sum(probs) / len(probs)  # 计算平均概率

            result = 1 if avg_prob > 0.5 else 0  # 1: AI合成, 0: 真实人脸
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
    parser.add_argument('--model_path', type=str, default='./src/best_model.pth', help='Path to the trained model')

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
    
    # 检查是否包含 'model_state_dict'
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    # 执行推理并保存结果
    infer_and_save_results(model, args.input_dir, args.output_csv, device)

if __name__ == '__main__':
    main()
