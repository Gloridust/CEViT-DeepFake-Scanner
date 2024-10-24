# main.py 用于执行推理并生成结果文件

import os
import pandas as pd
import argparse
import torch
from models import FinalModel
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def infer_and_save_results(model, input_dir, output_csv, device, threshold):
    # 定义基本的图像预处理
    transform = transforms.Compose([
        transforms.Resize((384, 384)),  # 确保使用正确的输入尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    print(f"扫描到的图片数量: {len(image_files)}")
    results = []

    # 使用 tqdm 显示进度条
    for filename in tqdm(image_files, desc='Processing images'):
        image_path = os.path.join(input_dir, filename)
        original_image = Image.open(image_path).convert('RGB')

        with torch.no_grad():
            image = transform(original_image).unsqueeze(0).to(device)
            output = model(image)
            prob = torch.softmax(output, dim=1)[:, 1].item()  # 获取 AI 生成的概率

            result = 1 if prob > threshold else 0
            results.append((os.path.splitext(filename)[0], result))

    # 保存结果到 CSV 文件
    results.sort()  # 按字典序排序
    df = pd.DataFrame(results, columns=['filename', 'result'])
    df.to_csv(output_csv, index=False, header=False)

def main():
    parser = argparse.ArgumentParser(description='AI-Generated Face Detection Inference')
    parser.add_argument('--input_dir', type=str, default='/testdata', help='Path to input directory containing images')
    parser.add_argument('--output_csv', type=str, default='./cla_pre.csv', help='Path to output CSV file')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'], help='Device to use for inference')
    parser.add_argument('--model_path', type=str, default='./src/checkpoints/best_model.pth', help='Path to the trained model')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for classification')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')

    args = parser.parse_args()

    # 设备检查
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

    # 打印模型信息
    print(f"Model loaded from: {args.model_path}")
    print(f"Running inference on device: {device}")
    print(f"Classification threshold: {args.threshold}")

    # 执行推理并保存结果
    infer_and_save_results(model, args.input_dir, args.output_csv, device, args.threshold)
    print(f"Results saved to: {args.output_csv}")

if __name__ == '__main__':
    main()
