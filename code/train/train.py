import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from tqdm import tqdm

from dataset import SVHNDataset
from model import SVHN_Model1


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def train(train_loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    # 使用tqdm显示进度条
    pbar = tqdm(train_loader, desc="Training")
    for i, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        c1, c2, c3, c4, c5, c6 = model(images)
        
        # 计算每个位置的损失
        loss = criterion(c1, labels[:, 0]) + \
               criterion(c2, labels[:, 1]) + \
               criterion(c3, labels[:, 2]) + \
               criterion(c4, labels[:, 3]) + \
               criterion(c5, labels[:, 4]) + \
               criterion(c6, labels[:, 5])
               
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss / (i + 1)})
        
    return running_loss / len(train_loader)

def validate(val_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 统计 0-10 每个类别的正确数量和总数量
    class_correct = torch.zeros(11, dtype=torch.long)
    class_total = torch.zeros(11, dtype=torch.long)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            c1, c2, c3, c4, c5, c6 = model(images)

            loss = criterion(c1, labels[:, 0]) + \
                   criterion(c2, labels[:, 1]) + \
                   criterion(c3, labels[:, 2]) + \
                   criterion(c4, labels[:, 3]) + \
                   criterion(c5, labels[:, 4]) + \
                   criterion(c6, labels[:, 5])

            running_loss += loss.item()

            # 计算竞赛得分：score = 编码识别正确数量 / 图片总数量
            # 这里要求 6 个位置全部预测正确才计为 1 张正确图片
            pred1 = c1.argmax(dim=1)
            pred2 = c2.argmax(dim=1)
            pred3 = c3.argmax(dim=1)
            pred4 = c4.argmax(dim=1)
            pred5 = c5.argmax(dim=1)
            pred6 = c6.argmax(dim=1)

            # 拼接预测结果和真实标签
            preds = torch.stack([pred1, pred2, pred3, pred4, pred5, pred6], dim=1)

            # 比较预测和真实标签是否完全一致
            correct += (preds == labels).all(dim=1).sum().item()
            total += labels.size(0)

            # 统计 class-wise 正确数量（0-10每个字符的准确率）
            # 使用向量化统计，避免循环中的频繁 .item() 导致 CPU/GPU 同步开销
            labels_flat = labels.view(-1).detach().cpu()
            correct_flat = (preds == labels).view(-1).detach().cpu()
            class_total += torch.bincount(labels_flat, minlength=11)
            class_correct += torch.bincount(labels_flat[correct_flat], minlength=11)

            pbar.set_postfix({'loss': running_loss / (i + 1), 'score': correct / total})

    score = correct / total
    
    # 计算每个类别的准确率，避免除以0
    class_acc = []
    for c in range(11):
        if class_total[c] > 0:
            class_acc.append(class_correct[c].item() / class_total[c].item())
        else:
            class_acc.append(0.0)
            
    return running_loss / len(val_loader), score, correct, total, class_acc

if __name__ == '__main__':
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03), # 数据增强
        transforms.RandomRotation(10), # 随机旋转
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # 随机平移
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet标准归一化
    ])


    val_transform = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 获取当前脚本所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tcdata_dir = os.path.abspath(os.path.join(current_dir, '../../tcdata'))
    
    # 3. 加载数据集
    train_dataset = SVHNDataset(
        img_path=os.path.join(tcdata_dir, 'mchar_train/mchar_train/'),
        img_label_json=os.path.join(tcdata_dir, 'mchar_train.json'),
        transform=train_transform
    )
    
    val_dataset = SVHNDataset(
        img_path=os.path.join(tcdata_dir, 'mchar_val/mchar_val/'),
        img_label_json=os.path.join(tcdata_dir, 'mchar_val.json'),
        transform=val_transform
    )
    
    # DataLoader 设置：在 Windows 上，子进程启动会造成每个 epoch 开头停顿。
    # 使用 persistent_workers=True 可避免每个 epoch 反复 spawn/kill worker（PyTorch>=1.7 支持）。
    # 同时降低 num_workers 到一个合理值，使用 prefetch_factor 控制每个 worker 的预取大小。
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # 预热 DataLoader：提前让 worker 创建并读取一批数据，避免训练开始时的长时间停顿
    try:
        print('Warming up DataLoader workers...')
        it = iter(train_loader)
        next(it)
    except Exception:
        # 如果数据读取出错（例如空数据集），忽略预热
        pass
    
    # 4. 初始化模型、损失函数和优化器
    model = SVHN_Model1().to(device)
    
    # 启用 cuDNN 自动调优，寻找最适合当前硬件的卷积算法
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        
    # 使用 Focal Loss + Label Smoothing，兼顾难样本与泛化能力
    # criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
    # 对照实验可切换为纯 CrossEntropyLoss（保留 label_smoothing）
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # 引入 L2 正则化 (Weight Decay)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
    # 引入学习率衰减 (LR Scheduler)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    
    # 5. 开始训练
    num_epochs = 20
    best_score = 0.0
    best_epoch = 0
    
    # 创建保存模型的文件夹
    user_data_dir = os.path.abspath(os.path.join(current_dir, '../../user_data/model_data'))
    os.makedirs(user_data_dir, exist_ok=True)

    # 创建日志目录（训练可视化）
    logs_dir = os.path.abspath(os.path.join(current_dir, '../../logs'))
    os.makedirs(logs_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(logs_dir, 'tb'))

    # 记录训练历史，便于保存曲线和结果对比
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_score': [],
        'val_correct': [],
        'val_total': [],
        'lr': []
    }
    # 动态添加 0-10 类别的准确率记录
    for c in range(11):
        history[f'cls{c}_acc'] = []
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        train_loss = train(train_loader, model, criterion, optimizer, device)
        val_loss, val_score, val_correct, val_total, class_acc = validate(val_loader, model, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Score: {val_score:.4f} ({val_correct}/{val_total})')
        
        # 打印所有 0-10 类别的准确率
        acc_str = ", ".join([f"{c}={class_acc[c]:.4f}" for c in range(11)])
        print(f'Class-wise Acc: {acc_str}')

        # TensorBoard 标量记录
        writer.add_scalar('Loss/Train', train_loss, epoch + 1)
        writer.add_scalar('Loss/Val', val_loss, epoch + 1)
        writer.add_scalar('Score/Val', val_score, epoch + 1)
        
        for c in range(11):
            writer.add_scalar(f'ClassAcc/{c}', class_acc[c], epoch + 1)
            history[f'cls{c}_acc'].append(class_acc[c])
            
        writer.add_scalar('LR', current_lr, epoch + 1)

        # 内存中保存训练历史
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_score'].append(val_score)
        history['val_correct'].append(val_correct)
        history['val_total'].append(val_total)
        history['lr'].append(current_lr)
        
        # 保存最好的模型
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(user_data_dir, 'best_model.pth'))
            print("Saved best model!")

    writer.close()

    # 保存训练历史 CSV
    history_df = pd.DataFrame(history)
    history_csv_path = os.path.join(logs_dir, 'train_history.csv')
    history_df.to_csv(history_csv_path, index=False)

    # 绘制并保存 Loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(logs_dir, 'loss_curve.png'), dpi=200)
    plt.close()

    # 绘制并保存 Val Score 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(history['epoch'], history['val_score'], label='Val Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(logs_dir, 'acc_curve.png'), dpi=200)
    plt.close()

    # 保存最佳模型信息
    best_info_path = os.path.join(logs_dir, 'best.txt')
    with open(best_info_path, 'w', encoding='utf-8') as f:
        f.write(f'best_epoch: {best_epoch}\n')
        f.write(f'best_val_score: {best_score:.6f}\n')

    print(f"Training history saved to: {history_csv_path}")
    print(f"Best model at epoch {best_epoch}, val_score={best_score:.6f}")
