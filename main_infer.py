# main_infer.py 用于推理（测试）模型的主程序

import argparse
import torch
from torch.utils.data import DataLoader
from models import FinalModel
from dataset import FaceDataset
from utils import inference

def main():
    parser = argparse.ArgumentParser(description='AI-Generated Face Detection Inference')
    parser.add_argument('--data_dir', type=str, default='data/test', help='Path to test data')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
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

    # 数据集和数据加载器
    test_dataset = FaceDataset(args.data_dir, train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 加载模型
    model = FinalModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # 开始推理
    predictions = inference(model, test_loader, device)
    print("Inference completed.")
    # 您可以在此处添加代码以保存或处理预测结果

if __name__ == '__main__':
    main()
