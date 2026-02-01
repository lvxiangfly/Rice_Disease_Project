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

# ================= ⚙️ 1. 专家知识矩阵 (必须与论文 Table 3 一致) =================
# 请核对您训练时的 ID 定义！这里是根据通常情况假设的 ID。
# 如果您的 Shape ID: 0是纺锤, 1是圆... 请保持。如果不一致，请修改这里的数字！

# 假设类别顺序: ['0_Blast', '1_BrownSpot', '2_Blight', '3_Tungro']
# Class ID -> { 'shape': Shape_ID, 'color': Color_ID, 'loc': Loc_ID }
EXPERT_RULES = {
    0: {'shape': 0, 'color': 0, 'loc': 0},  # Rice Blast (稻瘟) -> Spindle, Grey, Whole
    1: {'shape': 1, 'color': 0, 'loc': 0},  # Brown Spot (褐斑) -> Round, DarkBrown, Whole
    2: {'shape': 2, 'color': 2, 'loc': 1},  # Bacterial Blight (白叶枯) -> Irregular, YellowWhite, Tip
    3: {'shape': 2, 'color': 1, 'loc': 2}  # Tungro (东格鲁) -> Diffuse, YellowOrange, Whole
}
# 注意：如果有 'Healthy' 类 (ID 4)，且负样本被预测为 Healthy，通常 CS 很难定义。
# 策略：如果负样本被预测为 Healthy，且通过了 Bio-Filter，通常算作误报 (False Positive)。

# ================= ⚙️ 2. 基础配置 =================
BEST_H_MIN = 12
BEST_H_MAX = 65
TAU_ENV = 0.35
S_THRESHOLD = 60

RICE_TEST_DIR = r"D:\Rice_Disease_Project\Final_Dataset\test"
NOISE_TEST_DIR = r"D:\Rice_Disease_Project\test_noise"
MODEL_PATH = "multitask_best_model.pth"
CLASS_NAMES = ['0_Blast', '1_BrownSpot', '2_Blight', '3_Tungro']  # 请确认是否包含 Healthy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 🧠 模型定义 (保持不变) =================
class MultiTaskRiceNet(nn.Module):
    def __init__(self, num_classes, num_shapes, num_colors, num_locs):
        super(MultiTaskRiceNet, self).__init__()
        # 处理不同版本的 torchvision
        try:
            self.backbone = models.resnet50(weights=None)
        except:
            self.backbone = models.resnet50(pretrained=False)

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # 四个头
        self.cls_head = nn.Linear(num_ftrs, num_classes)
        self.shape_head = nn.Linear(num_ftrs, num_shapes)
        self.color_head = nn.Linear(num_ftrs, num_colors)
        self.loc_head = nn.Linear(num_ftrs, num_locs)

    def forward(self, x):
        features = self.backbone(x)
        return self.cls_head(features), self.shape_head(features), \
            self.color_head(features), self.loc_head(features)


def load_model():
    print(f"🔄 Loading MTRNet from: {MODEL_PATH} ...")
    model = MultiTaskRiceNet(
        num_classes=len(CLASS_NAMES),
        num_shapes=len(cfg.SHAPE_CLASSES),
        num_colors=len(cfg.COLOR_CLASSES),
        num_locs=len(cfg.LOC_CLASSES)
    )
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(DEVICE)
    model.eval()
    return model


# ================= 🧪 阶段 1: Bio-Grey Filter (HSV逻辑) =================
def bio_grey_filter(img_path):
    try:
        img_cv = cv2.imread(img_path)
        if img_cv is None: return False

        # Resize for speed
        h, w = img_cv.shape[:2]
        if w > 640:
            scale = 640 / w
            img_cv = cv2.resize(img_cv, (0, 0), fx=scale, fy=scale)
            h, w = img_cv.shape[:2]

        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # 论文公式: H in [12, 65], S > 60, V > 40
        mask_bio = (h_chan >= BEST_H_MIN) & (h_chan <= BEST_H_MAX) & \
                   (s_chan > S_THRESHOLD) & (v_chan > 40)

        ratio = np.count_nonzero(mask_bio) / (h * w)
        return ratio >= TAU_ENV  # 返回 True 表示是植物，通过过滤器
    except:
        return False


# ================= 🧪 阶段 2: Visual Consistency Check (VCC) =================
def calculate_consistency_score(pred_cls, pred_shape, pred_color, pred_loc):
    """
    计算一致性分数 (CS)
    :return: CS Score (0-3), is_consistent (bool)
    """
    cls_id = pred_cls.item()

    # 如果预测类别不在规则表中 (比如 Healthy 或其他)，根据具体情况处理
    if cls_id not in EXPERT_RULES:
        # 如果预测为 Healthy，通常认为没有特定形状，CS 逻辑不适用
        # 这里假设只处理 4 种病害
        return 0, False

    rule = EXPERT_RULES[cls_id]

    # 逐项比对
    match_shape = 1 if pred_shape.item() == rule['shape'] else 0
    match_color = 1 if pred_color.item() == rule['color'] else 0
    match_loc = 1 if pred_loc.item() == rule['loc'] else 0

    cs = match_shape + match_color + match_loc
    return cs, (cs == 3)  # 只有满分 3 才算完全一致 (或者 CS >= 2 也可以，看您论文怎么定，CS<3 即拒识)


# ================= 🚀 主测试流程 =================
def run_test(model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ---------------------------------------------------------
    # PART A: 正样本测试 (Acc, Confusion Matrix)
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("🚀 Part A: Positive Samples Evaluation (Accuracy)")
    print("=" * 50)

    y_true, y_pred = [], []

    for label_idx, class_name in enumerate(CLASS_NAMES):
        folder = os.path.join(RICE_TEST_DIR, class_name)
        if not os.path.exists(folder): continue

        files = [f for f in os.listdir(folder) if f.lower().endswith(('jpg', 'png'))]
        print(f"Testing {class_name} ({len(files)} images)...")

        for f in tqdm(files):
            path = os.path.join(folder, f)
            try:
                img = Image.open(path).convert('RGB')
                img_t = transform(img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    # 只用分类头计算准确率
                    out_cls, _, _, _ = model(img_t)
                    probs = F.softmax(out_cls, dim=1)
                    _, pred = torch.max(probs, 1)

                    y_true.append(label_idx)
                    y_pred.append(pred.item())
            except:
                pass

    acc = accuracy_score(y_true, y_pred)
    print(f"\n🏆 Test Accuracy: {acc * 100:.2f}%")

    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix (Acc: {acc * 100:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_final.png')
    print("✅ Matrix saved to confusion_matrix_final.png")

    # ---------------------------------------------------------
    # PART B: 负样本 OOD 测试 (Rejection Rate)
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("🚀 Part B: Negative Samples OOD Test (Bio-Filter + VCC)")
    print("=" * 50)

    noise_files = [os.path.join(NOISE_TEST_DIR, f) for f in os.listdir(NOISE_TEST_DIR)]
    total_noise = len(noise_files)
    rejected_count = 0

    # 统计具体是被谁拒识的
    rejected_by_bio = 0
    rejected_by_vcc = 0

    print(f"Testing {total_noise} negative samples...")

    for path in tqdm(noise_files):
        # --- Stage 1: Bio-Grey Filter ---
        if not bio_grey_filter(path):
            rejected_count += 1
            rejected_by_bio += 1
            continue  # 直接拒识，不进模型

        # --- Stage 2: Visual Consistency Check (VCC) ---
        try:
            img = Image.open(path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out_cls, out_shape, out_color, out_loc = model(img_t)

                # 获取所有头的预测 ID
                _, pred_cls = torch.max(out_cls, 1)
                _, pred_shape = torch.max(out_shape, 1)
                _, pred_color = torch.max(out_color, 1)
                _, pred_loc = torch.max(out_loc, 1)

                # 计算 CS 分数
                cs, is_consistent = calculate_consistency_score(
                    pred_cls, pred_shape, pred_color, pred_loc
                )

                # 论文逻辑: CS < 3 即为不一致 (拒识)
                if cs < 3:
                    rejected_count += 1
                    rejected_by_vcc += 1

        except Exception as e:
            print(e)
            pass

    orr = (rejected_count / total_noise) * 100
    print(f"\n🛡️ Final OOD Rejection Rate (ORR): {orr:.2f}%")
    print(f"   - Rejected by Bio-Filter: {rejected_by_bio}")
    print(f"   - Rejected by VCC Logic:  {rejected_by_vcc}")
    print(f"   - False Positives (Failed to reject): {total_noise - rejected_count}")


if __name__ == "__main__":
    model = load_model()
    run_test(model)