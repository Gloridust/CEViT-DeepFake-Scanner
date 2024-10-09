# main_infer.py 用于推理（分类）图片

import argparse
import torch
from torch.utils.data import DataLoader
from models import FinalModel
from torchvision import transforms
from PIL import Image
import os
import shutil
from tqdm import tqdm

def infer(model, image_path, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()
    
    return prob

def main():
    parser = argparse.ArgumentParser(description='AI-Generated Face Detection Inference')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input directory containing images')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory for classified images')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'], help='Device to use for inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU.")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS is not available. Switching to CPU.")
        args.device = 'cpu'

    device = torch.device(args.device)

    # 创建输出目录
    os.makedirs(os.path.join(args.output_dir, 'real'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'fake'), exist_ok=True)

    # 加载模型
    model = FinalModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # 获取所有图片文件
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # 处理输入目录中的所有图片
    for filename in tqdm(image_files, desc='Processing images'):
        image_path = os.path.join(args.input_dir, filename)
        prob = infer(model, image_path, transform, device)
        
        # 根据概率决定图片类别并移动到相应目录
        if prob > 0.5:
            shutil.copy(image_path, os.path.join(args.output_dir, 'real', filename))
        else:
            shutil.copy(image_path, os.path.join(args.output_dir, 'fake', filename))

    print("Inference completed. Images have been classified into 'real' and 'fake' folders.")

if __name__ == '__main__':
    main()