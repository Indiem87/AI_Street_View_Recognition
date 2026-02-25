import torch
import torch.nn as nn
import torchvision.models as models

class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()
        
        # 使用预训练的 ResNet34 作为特征提取器
        # 兼容 torchvision 旧/新版的 weights API：
        # 新版（>=0.13）应使用 weights=ResNet34_Weights.*，
        # 这里先尝试使用权重枚举，失败回退到 pretrained=True。
        try:
            from torchvision.models import ResNet34_Weights
            resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        except Exception:
            resnet = models.resnet34(pretrained=True)

        # 去掉最后的全连接层
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        
        # 定义6个分类头，每个头预测一个位置的字符
        # 类别数为11（0-9为数字，10为空白字符）
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)
        self.fc5 = nn.Linear(512, 11)
        self.fc6 = nn.Linear(512, 11)
        
    def forward(self, img):
        # 提取特征
        feat = self.cnn(img)
        # 展平特征图
        feat = feat.view(feat.shape[0], -1)
        feat = self.dropout(feat)
        
        # 分别预测6个位置的字符
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        c6 = self.fc6(feat)
        
        return c1, c2, c3, c4, c5, c6

# 测试代码
if __name__ == '__main__':
    model = SVHN_Model1()
    # 模拟输入一张图片 (Batch Size, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 64, 128)
    outputs = model(dummy_input)
    
    print("模型输出数量:", len(outputs))
    print("第一个位置的输出形状:", outputs[0].shape)
