import os
import argparse
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN  # 假设使用 facenet-pytorch 的 MTCNN 进行人脸检测

def preprocess_images(input_dir, output_dir, device):
    # 初始化MTCNN，人脸检测模型
    mtcnn = MTCNN(keep_all=True, device=device, post_process=False)  # 启用GPU加速，禁用后处理以提高速度

    # 定义预处理转换（仅用于后续裁剪后的图像）
    transform = transforms.Compose([
        transforms.Resize((384, 384)),  # 统一调整尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    print(f"扫描到的图片数量: {len(image_files)}")

    batch_size = 16  # 设置批处理大小，根据 GPU 显存调整
    resized_size = (384, 384)  # 统一调整尺寸

    for i in tqdm(range(0, len(image_files), batch_size), desc='预处理图片'):
        batch_files = image_files[i:i + batch_size]
        resized_images = []

        # 加载并统一调整尺寸的图像
        for f in batch_files:
            image_path = os.path.join(input_dir, f)
            try:
                image = Image.open(image_path).convert('RGB')
                image = image.resize(resized_size)  # 统一调整尺寸
                resized_images.append(image)
            except Exception as e:
                print(f"无法加载图像 {f}: {e}")
                resized_images.append(None)

        # 过滤掉加载失败的图像
        valid_images = [img for img in resized_images if img is not None]

        if not valid_images:
            continue  # 如果批次中没有有效图像，跳过

        # 批量检测人脸
        boxes, _ = mtcnn.detect(valid_images)

        for filename, original_image, box in zip(batch_files, resized_images, boxes):
            if original_image is None:
                continue  # 跳过无法加载的图像

            if box is not None:
                for j, single_box in enumerate(box):
                    # 计算裁剪区域
                    left, top, right, bottom = [int(coord) for coord in single_box]
                    cropped_face = original_image.crop((left, top, right, bottom))
                    # 保存裁剪后的人脸图像
                    base_name, ext = os.path.splitext(filename)
                    output_path = os.path.join(output_dir, f"{base_name}_face_{j}{ext}")
                    cropped_face.save(output_path)
            else:
                # 未检测到人脸时，复制原始图片
                original_output_path = os.path.join(output_dir, filename)
                original_image.save(original_output_path)

def main():
    parser = argparse.ArgumentParser(description='GPU人脸裁剪预处理')
    parser.add_argument('--dir', type=str, choices=['train', 'main'], default='main', help='指定预处理的目录类型')
    parser.add_argument('--input_dir', type=str, default='', help='输入图片目录（如果不指定，将根据--dir自动选择）')
    parser.add_argument('--output_dir', type=str, default='', help='预处理后图片的输出目录（如果不指定，将根据--dir自动选择）')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='使用的设备')

    args = parser.parse_args()

    if args.dir == 'train':
        input_dir = args.input_dir if args.input_dir else './data/train'
        output_dir = args.output_dir if args.output_dir else './data/train_processed'
        # 保留 real 和 fake 子目录
        for subset in ['real', 'fake']:
            subset_input = os.path.join(input_dir, subset)
            subset_output = os.path.join(output_dir, subset)
            os.makedirs(subset_output, exist_ok=True)
            preprocess_images(subset_input, subset_output, args.device)
    else:
        input_dir = args.input_dir if args.input_dir else '/testdata'
        output_dir = args.output_dir if args.output_dir else '/testdata_processed'
        os.makedirs(output_dir, exist_ok=True)
        preprocess_images(input_dir, output_dir, args.device)

if __name__ == '__main__':
    main()
