# main_infer_api.py 用于以API形式提供推理服务

import torch
from models import FinalModel
from torchvision import transforms
from PIL import Image

class FaceDetectionAPI:
    def __init__(self, model_path, device='cuda'):
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA is not available. Switching to CPU.")
            device = 'cpu'
        elif device == 'mps' and not torch.backends.mps.is_available():
            print("MPS is not available. Switching to CPU.")
            device = 'cpu'

        self.device = torch.device(device)
        self.model = FinalModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    def infer(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image)
            prob = torch.sigmoid(output).item()
        
        # 0: AI生成人脸, 1: 真实人脸
        result = 1 if prob > 0.5 else 0
        
        return result, prob

# 使用示例
if __name__ == '__main__':
    model_path = 'path/to/your/model.pth'
    api = FaceDetectionAPI(model_path)
    
    image_path = 'path/to/your/image.jpg'
    result, probability = api.infer(image_path)
    
    print(f"Classification result: {result}")  # 0: AI生成人脸, 1: 真实人脸
    print(f"Probability of being a real face: {probability:.4f}")