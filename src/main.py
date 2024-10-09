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
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    results = []

    for filename in tqdm(image_files, desc='Processing images'):
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            prob = torch.sigmoid(output).item()
            result = 1 if prob <= 0.5 else 0  # 0: AI生成, 1: 真实人脸
            results.append((os.path.splitext(filename)[0], result))  # 保存文件名（无扩展名）和结果

    # 保存结果到 CSV 文件
    results.sort()  # 按字典序排序
    df = pd.DataFrame(results, columns=['filename', 'result'])
    df.to_csv(output_csv, index=False, header=False)

def main():
    parser = argparse.ArgumentParser(description='AI-Generated Face Detection Inference')
    parser.add_argument('--input_dir', type=str, default='./testdata', help='Path to input directory containing images')
    parser.add_argument('--output_csv', type=str, default='./cla_pre.csv', help='Path to output CSV file')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'], help='Device to use for inference')
    parser.add_argument('--model_path', type=str, default='./src/checkpoint_epoch_30.pth', help='Path to the trained model')

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
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 执行推理并保存结果
    infer_and_save_results(model, args.input_dir, args.output_csv, device)

if __name__ == '__main__':
    main()