import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import copy
from tqdm import tqdm
import attributes_config1 as cfg  # 👈 导入您的配置文件

# ================= ⚙️ 配置区域 =================
# 数据集路径 (指向您刚刚划分好的 Final_Dataset)
DATA_DIR = r"D:\Rice_Disease_Project\Final_Dataset"

# 训练参数
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50  # 建议跑 50-100 轮
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 权重设置 (可以调整辅助任务的重要性)
LAMBDA_CLS = 1.0  # 主任务权重
LAMBDA_SHAPE = 0.5  # 形状头权重
LAMBDA_COLOR = 0.5  # 颜色头权重
LAMBDA_LOC = 0.5  # 位置头权重


# ===============================================

# === 1. 定义多任务网络结构 (Multi-Task RiceNet) ===
class MultiTaskRiceNet(nn.Module):
    def __init__(self, num_classes, num_shapes, num_colors, num_locs):
        super(MultiTaskRiceNet, self).__init__()

        # 加载 ResNet50 骨干
        try:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except:
            self.backbone = models.resnet50(pretrained=True)

        # 获取特征维度 (2048)
        num_ftrs = self.backbone.fc.in_features

        # 去掉原始的全连接层
        self.backbone.fc = nn.Identity()

        # === 定义 4 个独立的头 ===
        # 1. 主分类头 (病害类别)
        self.cls_head = nn.Linear(num_ftrs, num_classes)
        # 2. 形状头
        self.shape_head = nn.Linear(num_ftrs, num_shapes)
        # 3. 颜色头
        self.color_head = nn.Linear(num_ftrs, num_colors)
        # 4. 位置头
        self.loc_head = nn.Linear(num_ftrs, num_locs)

    def forward(self, x):
        # 提取公共特征 [Batch, 2048]
        features = self.backbone(x)

        # 各个头独立预测
        y_cls = self.cls_head(features)
        y_shape = self.shape_head(features)
        y_color = self.color_head(features)
        y_loc = self.loc_head(features)

        return y_cls, y_shape, y_color, y_loc


# === 2. 训练主程序 ===
def train_multitask():
    print(f"🚀 启动 Multi-Task 训练...")
    print(f"   利用 attributes_config.py 进行知识注入")

    # --- 数据预处理 ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # --- 加载数据 ---
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes  # e.g. ['Blast', 'Brown', ...]

    print(f"📊 类别列表: {class_names}")

    # --- 初始化模型 ---
    model = MultiTaskRiceNet(
        num_classes=len(class_names),
        num_shapes=len(cfg.SHAPE_CLASSES),
        num_colors=len(cfg.COLOR_CLASSES),
        num_locs=len(cfg.LOC_CLASSES)
    )
    model = model.to(DEVICE)

    # --- 优化器与损失 ---
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # --- 训练循环 ---
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch + 1}/{NUM_EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 进度条
            pbar = tqdm(dataloaders[phase], desc=f"{phase}")

            for inputs, labels in pbar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)  # 这是病害标签 (0,1,2,3)

                # ================= 🗝️ 核心：从 Config 生成辅助标签 =================
                # 我们需要在 GPU 上构建形状、颜色、位置的标签
                batch_size = labels.size(0)
                shape_lbls = []
                color_lbls = []
                loc_lbls = []

                for lbl in labels:
                    # 调用 config 里的函数，把病害ID转成属性ID
                    s, c, l = cfg.get_attr_labels_by_index(lbl.item(), class_names)
                    shape_lbls.append(s)
                    color_lbls.append(c)
                    loc_lbls.append(l)

                # 转为 Tensor 并移到 GPU
                target_shape = torch.tensor(shape_lbls).to(DEVICE)
                target_color = torch.tensor(color_lbls).to(DEVICE)
                target_loc = torch.tensor(loc_lbls).to(DEVICE)
                # ================================================================

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # 前向传播：出 4 个结果
                    out_cls, out_shape, out_color, out_loc = model(inputs)

                    # 获取主任务预测结果
                    _, preds = torch.max(out_cls, 1)

                    # 计算 4 个 Loss
                    l_cls = criterion(out_cls, labels)
                    l_shape = criterion(out_shape, target_shape)
                    l_color = criterion(out_color, target_color)
                    l_loc = criterion(out_loc, target_loc)

                    # 加权求和
                    total_loss = (LAMBDA_CLS * l_cls) + \
                                 (LAMBDA_SHAPE * l_shape) + \
                                 (LAMBDA_COLOR * l_color) + \
                                 (LAMBDA_LOC * l_loc)

                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()

                # 统计 (我们主要看主任务 Accuracy)
                running_loss += total_loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 进度条显示 Loss
                pbar.set_postfix({'Total Loss': total_loss.item()})

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # 保存时，名字叫 multitask_best.pth 以示区分
                torch.save(model.state_dict(), 'multitask_best_model.pth')
                print(f"✨ 新最佳模型 (Acc: {best_acc:.4f}) 已保存！")

    print("🏁 Multi-Task 训练完成！")


if __name__ == "__main__":
    train_multitask()