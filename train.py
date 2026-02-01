import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# 导入你自己写的模型和配置
from model import MultiTaskRiceNet
from attributes_config import ATTRIBUTE_GT

# ================= 配置区域 =================
# 【重要】请修改为你存放图片文件夹的实际路径
# 路径里不要有中文，文件夹必须是 0_Blast, 1_BrownSpot 这种格式
DATA_DIR = r"D:\Rice_Disease_Project\data"

BATCH_SIZE = 16  # 每次喂给模型多少张图
LEARNING_RATE = 0.001  # 学习率
NUM_EPOCHS = 10  # 训练几轮 (演示用10轮，写论文建议30-50)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    print(f"正在使用设备: {DEVICE}")

    # 1. 图像预处理 (缩放、转Tensor、归一化)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. 读取数据
    if not os.path.exists(DATA_DIR):
        print(f"❌ 错误：找不到文件夹 {DATA_DIR}，请检查路径！")
        return

    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=data_transforms)
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"✅ 成功加载数据！共 {len(full_dataset)} 张图片。")
    print(f"类别映射: {full_dataset.class_to_idx}")
    # 确保打印出来是 {'0_Blast': 0, '1_BrownSpot': 1 ...}

    # 3. 初始化模型
    model = MultiTaskRiceNet().to(DEVICE)

    # 4. 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 5. 开始训练循环
    print("🚀 开始训练三头蛇模型...")
    model.train()  # 切换到训练模式

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0

        for images, disease_labels in train_loader:
            images = images.to(DEVICE)
            disease_labels = disease_labels.to(DEVICE)

            # --- 核心：动态生成属性标签 (Ground Truth) ---
            # 根据病害ID (例如0)，去字典里查它的形状、颜色、位置属性
            shape_targets = []
            color_targets = []
            loc_targets = []

            for label in disease_labels:
                lbl_idx = label.item()
                attrs = ATTRIBUTE_GT[lbl_idx]  # 查字典
                shape_targets.append(attrs['shape'])
                color_targets.append(attrs['color'])
                loc_targets.append(attrs['loc'])

            # 转为 Tensor
            shape_targets = torch.tensor(shape_targets).to(DEVICE)
            color_targets = torch.tensor(color_targets).to(DEVICE)
            loc_targets = torch.tensor(loc_targets).to(DEVICE)

            # --- 前向传播 ---
            optimizer.zero_grad()  # 清空梯度

            # 模型同时输出三个头的结果
            pred_shape, pred_color, pred_loc = model(images)

            # --- 计算多任务损失 ---
            loss_shape = criterion(pred_shape, shape_targets)
            loss_color = criterion(pred_color, color_targets)
            loss_loc = criterion(pred_loc, loc_targets)

            # 总损失 = 三个损失之和
            total_loss = loss_shape + loss_color + loss_loc

            # --- 反向传播 ---
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        # 打印这一轮的训练情况
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Total Loss: {epoch_loss:.4f}")

    # 6. 保存模型
    torch.save(model.state_dict(), "rice_multitask_model.pth")
    print("💾 模型已保存为 rice_multitask_model.pth")


if __name__ == '__main__':
    train()