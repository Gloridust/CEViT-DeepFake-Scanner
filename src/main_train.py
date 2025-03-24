# main_train.py 用于训练模型的主程序

import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from models import FinalModel
from dataset import FaceDataset
from utils import train, validate  # 移除 FocalLoss 导入
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import random
import numpy as np
import time
import os

def main():
    # 在 main() 函数开始处添加日志初始化
    parser = argparse.ArgumentParser(description='AI-Generated Face Detection Training')
    parser.add_argument('--data_dir', type=str, default='data/train', help='Path to training data')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'], help='Device to use for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')  # 添加这行

    args = parser.parse_args()

    device = torch.device(args.device)

    # 根据设备类型决定是否使用混合精度
    if device.type == 'cuda':
        scaler = GradScaler()
    else:
        scaler = None

    # 数据集和数据加载器
    total_dataset = FaceDataset(args.data_dir, train=True)
    train_size = int(0.8 * len(total_dataset))
    val_size = len(total_dataset) - train_size
    train_dataset, val_dataset = random_split(total_dataset, [train_size, val_size])

    # 计算训练集中的正负样本数量
    train_labels = [total_dataset.labels[i] for i in train_dataset.indices]
    val_labels = [total_dataset.labels[i] for i in val_dataset.indices]

    num_train_positive = sum(train_labels)
    num_train_negative = len(train_labels) - num_train_positive
    num_val_positive = sum(val_labels)
    num_val_negative = len(val_labels) - num_val_positive

    # 打印训练和验证集的样本数量
    print(f"训练集样本数量: 正样本={num_train_positive}, 负样本={num_train_negative}")
    print(f"验证集样本数量: 正样本={num_val_positive}, 负样本={num_val_negative}")

    # 计算类别权重
    total = num_train_positive + num_train_negative
    weight0 = total / (2 * num_train_negative)
    weight1 = total / (2 * num_train_positive)
    class_weights = torch.tensor([weight0, weight1], device=device)

    # 创建样本权重
    class_sample_counts = np.bincount(train_labels)
    weights = 1. / class_sample_counts
    samples_weights = np.array([weights[label] for label in train_labels])
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    # 使用加权的 CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 使用 WeightedRandomSampler 创建训练数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # 定义训练阶段
    num_epochs_phase1 = 5  # 第一阶段：只训练融合层
    num_epochs_phase2 = args.epochs - num_epochs_phase1  # 第二阶段：微调整个模型
    
    # 初始化模型
    model = FinalModel().to(device)
    start_epoch = 1
    best_val_loss = np.inf
    best_val_auc = 0.0
    trigger_times = 0
    
    # 初始化 checkpoint 变量，避免后面未定义的错误
    checkpoint = None

    # 加载检查点
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"加载检查点 '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', np.inf)
            best_val_auc = checkpoint.get('best_val_auc', 0.0)
            trigger_times = checkpoint.get('trigger_times', 0)
            print(f"检查点 '{args.resume}' (epoch {checkpoint['epoch']}) 已加载")

    # 根据当前epoch确定训练阶段
    if start_epoch <= num_epochs_phase1:
        print("当前为第一阶段：训练特征融合层")
        model.freeze_base_models()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_phase1, eta_min=1e-6)
        
        # 训练第一阶段剩余的epoch
        for epoch in range(start_epoch, num_epochs_phase1 + 1):
            start_time = time.time()  # 记录epoch开始时间

            train_loss = train(model, train_loader, criterion, optimizer, device, scaler)
            val_loss, accuracy, precision, recall, f1, val_auc = validate(model, val_loader, criterion, device)  # 修改 validate 以返回 val_auc

            # 更新学习率
            scheduler.step()

            epoch_duration = time.time() - start_time  # 计算epoch时长

            print(f"Epoch [{epoch}/{num_epochs_phase1}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
                  f"F1 Score: {f1:.4f}, ROC-AUC: {val_auc:.4f}, Duration: {epoch_duration:.2f}s")

            # 保存最佳模型
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                trigger_times = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_auc': best_val_auc,
                    'trigger_times': trigger_times
                }, './src/checkpoints/best_model.pth')
                print("保存当前最佳模型 (基于 ROC-AUC)")
            else:
                trigger_times += 1
                print(f"早停计数器：{trigger_times}")
                if trigger_times >= args.patience:  # 修改这里
                    print("触发早停，停止训练")
                    break

            # 记录每个 epoch 的验证信息
            with open('./src/training_log.txt', 'a') as log_file:
                log_file.write(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                               f"Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, "
                               f"F1 Score={f1:.4f}, ROC-AUC={val_auc:.4f}, Duration={epoch_duration:.2f}s\n")

            # 保存一个检查点，以便之后恢复
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_auc': best_val_auc,
                'trigger_times': trigger_times
            }, f'./src/checkpoints/checkpoint_epoch_{epoch}.pth')

        # 完成第一阶段后，自动进入第二阶段
        start_epoch = num_epochs_phase1 + 1
    
    # 第二阶段：全模型微调
    if start_epoch > num_epochs_phase1:
        print("当前为第二阶段：微调全模型")
        model.unfreeze_base_models()
        optimizer = optim.Adam([
            {'params': model.fusion.parameters(), 'lr': args.lr},
            {'params': model.convnext.parameters(), 'lr': args.lr * 0.1},
            {'params': model.efficientnet.parameters(), 'lr': args.lr * 0.1},
            {'params': model.vit.parameters(), 'lr': args.lr * 0.1}
        ])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_phase2, eta_min=1e-7)
        
        # 训练第二阶段
        for epoch in range(start_epoch, args.epochs + 1):
            start_time = time.time()  # 记录epoch开始时间

            train_loss = train(model, train_loader, criterion, optimizer, device, scaler)
            val_loss, accuracy, precision, recall, f1, val_auc = validate(model, val_loader, criterion, device)  # 修改 validate 以返回 val_auc

            # 更新学习率
            scheduler.step()

            epoch_duration = time.time() - start_time  # 计算epoch时长

            print(f"Epoch [{epoch}/{args.epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
                  f"F1 Score: {f1:.4f}, ROC-AUC: {val_auc:.4f}, Duration: {epoch_duration:.2f}s")

            # 保存最佳模型
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                trigger_times = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_auc': best_val_auc,
                    'trigger_times': trigger_times
                }, './src/checkpoints/best_model.pth')
                print("保存当前最佳模型 (基于 ROC-AUC)")
            else:
                trigger_times += 1
                print(f"早停计数器：{trigger_times}")
                if trigger_times >= args.patience:  # 修改这里
                    print("触发早停，停止训练")
                    break

            # 记录每个 epoch 的验证信息
            with open('./src/training_log.txt', 'a') as log_file:
                log_file.write(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                               f"Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, "
                               f"F1 Score={f1:.4f}, ROC-AUC={val_auc:.4f}, Duration={epoch_duration:.2f}s\n")

            # 保存一个检查点，以便之后恢复
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_auc': best_val_auc,
                'trigger_times': trigger_times
            }, f'./src/checkpoints/checkpoint_epoch_{epoch}.pth')

    try:
        # 尝试加载优化器状态
        if checkpoint is not None and 'optimizer_state_dict' in checkpoint and 'scheduler_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("成功加载优化器和调度器状态")
    except (ValueError, KeyError) as e:
        print(f"警告: 无法加载优化器或调度器状态，将使用新的初始化状态。错误: {str(e)}")

    # 在 main() 函数开始处添加日志初始化
    # 创建日志文件，记录训练配置
    with open('./src/training_log.txt', 'a') as log_file:
        log_file.write(f"\n{'='*50}\n")
        log_file.write(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Model architecture: ConvNext-Base + EfficientNet-B2 + ViT-Small\n")
        log_file.write(f"Training parameters:\n")
        log_file.write(f"- Batch size: {args.batch_size}\n")
        log_file.write(f"- Learning rate: {args.lr}\n")
        log_file.write(f"- Device: {args.device}\n")
        log_file.write(f"- Total epochs: {args.epochs}\n")
        log_file.write(f"- Phase 1 epochs: {num_epochs_phase1}\n")
        log_file.write(f"- Phase 2 epochs: {num_epochs_phase2}\n")
        log_file.write(f"Dataset information:\n")
        log_file.write(f"- Training samples: {len(train_dataset)} (Positive: {num_train_positive}, Negative: {num_train_negative})\n")
        log_file.write(f"- Validation samples: {len(val_dataset)} (Positive: {num_val_positive}, Negative: {num_val_negative})\n")
        log_file.write(f"{'='*50}\n\n")

    # 在每个 epoch 的日志记录中添加更多信息
    with open('./src/training_log.txt', 'a') as log_file:
        log_file.write(f"Epoch {epoch} ({time.strftime('%H:%M:%S')}):\n")
        log_file.write(f"- Phase: {'Fusion training' if epoch <= num_epochs_phase1 else 'Fine-tuning'}\n")
        log_file.write(f"- Learning rates: {[param_group['lr'] for param_group in optimizer.param_groups]}\n")
        log_file.write(f"- Train Loss: {train_loss:.4f}\n")
        log_file.write(f"- Val Loss: {val_loss:.4f}\n")
        log_file.write(f"- Accuracy: {accuracy:.4f}\n")
        log_file.write(f"- Precision: {precision:.4f}\n")
        log_file.write(f"- Recall: {recall:.4f}\n")
        log_file.write(f"- F1 Score: {f1:.4f}\n")
        log_file.write(f"- ROC-AUC: {val_auc:.4f}\n")
        log_file.write(f"- Duration: {epoch_duration:.2f}s\n")
        log_file.write(f"- Best val AUC so far: {best_val_auc:.4f}\n")
        log_file.write(f"- Early stopping counter: {trigger_times}/{args.patience}\n\n")

if __name__ == '__main__':
    main()
