import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class MultiTaskRiceNet(nn.Module):
    def __init__(self):
        super(MultiTaskRiceNet, self).__init__()

        # 1. 加载预训练的 ResNet50 (作为特征提取器 Backbone)
        # 使用最新的 weights 参数替代旧版 pretrained=True
        weights = ResNet50_Weights.DEFAULT
        self.backbone = resnet50(weights=weights)

        # 获取全连接层之前的特征维度 (ResNet50 是 2048)
        in_features = self.backbone.fc.in_features

        # 去掉原始的全连接层
        self.backbone.fc = nn.Identity()

        # 2. 创新点：设计三个独立的“分类头” (Heads)
        # Head 1: 识别形状 (4类: 纺锤/圆/波浪/全叶)
        self.shape_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4)
        )

        # Head 2: 识别颜色 (3类: 灰白/褐/橙黄)
        self.color_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3)
        )

        # Head 3: 识别位置 (2类: 随机/叶尖)
        self.loc_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        # 提取公共特征 (Common Features)
        features = self.backbone(x)

        # 多任务输出
        shape_out = self.shape_head(features)
        color_out = self.color_head(features)
        loc_out = self.loc_head(features)

        return shape_out, color_out, loc_out


# 测试一下模型通不通
if __name__ == "__main__":
    model = MultiTaskRiceNet()
    dummy_img = torch.randn(2, 3, 224, 224)  # 模拟2张图片
    s, c, l = model(dummy_img)
    print("Shape output:", s.shape)  # 应为 [2, 4]
    print("Color output:", c.shape)  # 应为 [2, 3]
    print("Location output:", l.shape)  # 应为 [2, 2]
    print("模型构建成功！")