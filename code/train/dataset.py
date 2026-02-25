import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label_json, transform=None):
        """
        img_path: 图片文件夹路径
        img_label_json: 对应的json标注文件路径
        transform: 数据增强/预处理操作
        """
        self.img_path = img_path
        self.transform = transform
        
        # 读取json文件
        with open(img_label_json, 'r') as f:
            self.img_label = json.load(f)
            
        # 获取所有的图片文件名
        self.img_names = list(self.img_label.keys())
        
    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_full_path = os.path.join(self.img_path, img_name)
        
        # 读取图片并转换为RGB格式
        img = Image.open(img_full_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        # 获取标签
        label = self.img_label[img_name]['label']
        
        # 确保 label 是一个列表 (因为有些图片可能只有一个字符，JSON 解析出来可能是整数)
        if not isinstance(label, list):
            label = [label]
        
        # 填充标签到固定长度（比如最长6个字符）
        # 10 表示空白字符（没有数字）
        lbl_pad = [10] * 6
        for i in range(min(len(label), 6)):
            lbl_pad[i] = label[i]
            
        return img, torch.tensor(lbl_pad, dtype=torch.long)
        
    def __len__(self):
        return len(self.img_names)

# 测试代码
if __name__ == '__main__':
    # 获取当前脚本所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接出 tcdata 目录的绝对路径
    tcdata_dir = os.path.abspath(os.path.join(current_dir, '../../tcdata'))
    
    # 定义预处理操作：缩放到固定大小并转换为Tensor
    transform = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ToTensor(),
    ])
    
    # 实例化数据集
    train_dataset = SVHNDataset(
        img_path=os.path.join(tcdata_dir, 'mchar_train/mchar_train/'),
        img_label_json=os.path.join(tcdata_dir, 'mchar_train.json'),
        transform=transform
    )
    
    # 测试读取第一个样本
    img, label = train_dataset[0]
    print("图片大小:", img.shape)
    print("标签:", label)
