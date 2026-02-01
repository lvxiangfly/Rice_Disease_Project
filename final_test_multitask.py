import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import os
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import attributes_config1 as cfg  # 导入您的配置

# ================= ⚙️ 配置区域 =================
# 1. 最佳阈值 (请确认这是您用 Final_Dataset/train 重新算出来的，或者沿用之前的)
BEST_H_MIN = 12
BEST_H_MAX = 65
TAU_ENV = 0.25
S_THRESHOLD = 60

# 2. 路径
RICE_TEST_DIR = r"D:\Rice_Disease_Project\Final_Dataset\test"  # 👈 必须是新划分的测试集
NOISE_TEST_DIR = r"D:\Rice_Disease_Project\test_noise"  # 负样本路径
MODEL_PATH = "multitask_best_model.pth"  # 👈 必须是新训练的多任务模型

# 3. 类别定义
CLASS_NAMES = ['0_Blast', '1_BrownSpot', '2_Blight', '3_Tungro']  # 请按训练时的字母顺序排列!
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================================

# 定义模型结构 (必须与训练时一致)
class MultiTaskRiceNet(nn.Module):
    def __init__(self, num_classes, num_shapes, num_colors, num_locs):
        super(MultiTaskRiceNet, self).__init__()
        try:
            self.backbone = models.resnet50(weights=None)
        except:
            self.backbone = models.resnet50(pretrained=False)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.cls_head = nn.Linear(num_ftrs, num_classes)
        self.shape_head = nn.Linear(num_ftrs, num_shapes)
        self.color_head = nn.Linear(num_ftrs, num_colors)
        self.loc_head = nn.Linear(num_ftrs, num_locs)

    def forward(self, x):
        features = self.backbone(x)
        return self.cls_head(features), self.shape_head(features), self.color_head(features), self.loc_head(features)


def load_model():
    print(f"🔄 正在加载多任务模型: {MODEL_PATH} ...")
    model = MultiTaskRiceNet(
        num_classes=len(CLASS_NAMES),
        num_shapes=len(cfg.SHAPE_CLASSES),
        num_colors=len(cfg.COLOR_CLASSES),
        num_locs=len(cfg.LOC_CLASSES)
    )
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint, strict=False)  # 兼容
    model = model.to(DEVICE)
    model.eval()
    return model


def bio_grey_filter(img_path):
    """ 生物灰度滤波器 """
    try:
        img_cv = cv2.imread(img_path)
        if img_cv is None: return False
        h, w = img_cv.shape[:2]
        if w > 640:
            scale = 640 / w
            img_cv = cv2.resize(img_cv, (0, 0), fx=scale, fy=scale)
            h, w = img_cv.shape[:2]
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        mask_bio = (h_chan >= BEST_H_MIN) & (h_chan <= BEST_H_MAX) & \
                   (s_chan > S_THRESHOLD) & (v_chan > 40)

        ratio = np.count_nonzero(mask_bio) / (h * w)
        return ratio >= TAU_ENV
    except:
        return False


def run_test(model):
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("\n" + "=" * 50)
    print("🚀 阶段 1: 水稻测试集评估 (Accuracy & Confusion Matrix)")
    print("=" * 50)

    y_true = []
    y_pred = []

    # 遍历测试集子文件夹
    total_imgs = 0

    # 按照 CLASS_NAMES 的顺序遍历，确保 Label ID 对应
    for label_idx, class_name in enumerate(CLASS_NAMES):
        folder = os.path.join(RICE_TEST_DIR, class_name)
        if not os.path.exists(folder):
            print(f"⚠️ 警告: 找不到文件夹 {folder}")
            continue

        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
        print(f"   正在测试类别: {class_name} ({len(files)}张)...")

        for f in tqdm(files):
            total_imgs += 1
            path = os.path.join(folder, f)

            # 1. 滤波器 (Pipeline Step 1)
            # 虽然算 Accuracy 时通常只看模型，但为了模拟真实情况，
            # 如果被滤波器过滤了，我们暂且算作“识别错误”或者“拒识”
            # 这里为了画混淆矩阵，我们只统计“通过滤波器”的样本，或者全部统计。
            # 策略：全部统计，看 Raw Accuracy。

            try:
                img = Image.open(path).convert('RGB')
                img_t = transform(img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    # 关键修改：接收 4 个输出，只用第 1 个
                    out_cls, _, _, _ = model(img_t)
                    probs = F.softmax(out_cls, dim=1)
                    _, pred = torch.max(probs, 1)

                    y_true.append(label_idx)
                    y_pred.append(pred.item())
            except Exception as e:
                print(f"Error: {f} {e}")

    # --- 计算指标 ---
    acc = accuracy_score(y_true, y_pred)
    print(f"\n🏆 水稻测试集总准确率 (Raw Accuracy): {acc * 100:.2f}%")
    print("\n📄 详细分类报告:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # --- 画混淆矩阵 ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (MTRNet)')
    plt.tight_layout()
    plt.savefig('confusion_matrix_final.png', dpi=300)
    print("✅ 混淆矩阵已保存为 confusion_matrix_final.png")

    print("\n" + "=" * 50)
    print("🚀 阶段 2: 负样本 OOD 测试 (拒识率)")
    print("=" * 50)

    noise_files = [os.path.join(NOISE_TEST_DIR, f) for f in os.listdir(NOISE_TEST_DIR)]
    rejected_count = 0
    total_noise = len(noise_files)

    print(f"正在测试 {total_noise} 张负样本...")

    for path in tqdm(noise_files):
        # 1. Bio-Grey Filter
        if not bio_grey_filter(path):
            rejected_count += 1
            continue

        # 2. VCC Check (模拟)
        # 这里是一个简单的 VCC 逻辑模拟：
        # 如果模型对其非常不自信 (Conf < 0.7)，或者属性预测非常离谱，也可以拒识
        # 为了简化实验，暂时只统计 Filter 的拒识率，
        # 如果您想体现 VCC，可以加上置信度阈值

        try:
            img = Image.open(path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out_cls, _, _, _ = model(img_t)
                probs = F.softmax(out_cls, dim=1)
                conf, _ = torch.max(probs, 1)

                # 假设 VCC 逻辑：如果置信度不高，也算拒识
                if conf.item() < 0.8:  # VCC 阈值
                    rejected_count += 1
        except:
            pass

    orr = (rejected_count / total_noise) * 100
    print(f"\n🛡️ 最终 OOD 拒识率 (Filter + VCC): {orr:.2f}%")


if __name__ == "__main__":
    model = load_model()
    run_test(model)