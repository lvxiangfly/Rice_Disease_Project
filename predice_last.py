import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import os
import cv2
import numpy as np
import glob

# 引入你的模型结构
from model import MultiTaskRiceNet
from attributes_config import ATTRIBUTE_GT, DISEASE_MAPPING

# ================= ⚙️ 论文级参数配置 =================
MODEL_PATH = "rice_multitask_model.pth"
TEST_DIR = r"D:\Rice_Disease_Project\test_images"

# 1. 基础置信度 (0.70)
# 论文中可以称之为: "Confidence Thresholding Strategy"
CONFIDENCE_THRESHOLD = 0.70

# 2. 绿色阈值 (5%)
# 论文中可以称之为: "Background Chlorophyll Filter"
MIN_GREEN_RATIO = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 属性文字
SHAPE_TEXT = {0: "Spindle", 1: "Round", 2: "Irregular", 3: "Full"}
COLOR_TEXT = {0: "Gray/White", 1: "Brown", 2: "Orange"}
LOC_TEXT = {0: "Random", 1: "Tip/Edge"}


# ================= 🛡️ 核心检测逻辑 =================

def check_environment_validity(img_bgr):
    """
    环境检测器：负责拦截非农业环境图像（如猫、室内家具）
    """
    h, w = img_bgr.shape[:2]
    total = h * w

    # 转 HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 1. 计算绿色占比 (生命力特征)
    # H: 30-90, S: >30
    mask_green = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([90, 255, 255]))
    green_ratio = np.count_nonzero(mask_green) / total

    # 2. 拦截逻辑
    if green_ratio < MIN_GREEN_RATIO:
        return False, f"Low Chlorophyll ({green_ratio:.1%})"

    return True, f"Valid ({green_ratio:.1%})"


def load_model():
    model = MultiTaskRiceNet()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    return None


def predict_single_image(image_path, model, transform):
    filename = os.path.basename(image_path)
    img_cv = cv2.imread(image_path)

    if img_cv is None:
        print(f"❌ 读取失败: {filename}")
        return

    # --- Step 1: 环境预筛 (论文中的 Pre-processing) ---
    is_valid_env, env_msg = check_environment_validity(img_cv)
    if not is_valid_env:
        print(f"🛑 [OOD拒识] {filename} -> 非水稻环境: {env_msg}")
        # 在统计时，这算作“成功拦截负样本”
        return

    # --- Step 2: 模型推理 (Deep Learning Inference) ---
    try:
        img_pil = Image.open(image_path).convert('RGB')
        img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    except:
        return

    with torch.no_grad():
        logit_shape, logit_color, logit_loc = model(img_tensor)

        # 计算 Softmax 概率
        prob_shape = F.softmax(logit_shape, dim=1)
        prob_color = F.softmax(logit_color, dim=1)
        prob_loc = F.softmax(logit_loc, dim=1)

        # 获取最大置信度
        c_shape, i_shape = torch.max(prob_shape, 1)
        c_color, i_color = torch.max(prob_color, 1)
        c_loc, i_loc = torch.max(prob_loc, 1)

        # 木桶效应：取最低分
        min_conf = min(c_shape.item(), c_color.item(), c_loc.item())

        # 获取 ID
        pred_color_id = i_color.item()
        pred_shape_id = i_shape.item()
        pred_loc_id = i_loc.item()

    # --- Step 3: 置信度门控 (Confidence Gating) ---
    if min_conf < CONFIDENCE_THRESHOLD:
        print(f"⚠️ [不确定] {filename} -> 置信度不足 ({min_conf:.2f})")
        return

    # --- Step 4: 逻辑映射 (Knowledge Mapping) ---
    max_score = 0
    best_match = "Unknown"
    for disease_id, rules in ATTRIBUTE_GT.items():
        score = 0
        if rules['shape'] == pred_shape_id: score += 1
        if rules['color'] == pred_color_id: score += 1
        if rules['loc'] == pred_loc_id:   score += 1
        if score > max_score:
            max_score = score
            best_match = DISEASE_MAPPING[disease_id]

    # --- 最终输出 ---
    # 如果花盆依然被识别，但置信度很高，那就是模型的 Limitation
    # 此时不要强行用代码修，而是记录下来作为 Discussion 素材
    print(f"✅ [确诊] {filename:<15} -> 【{best_match}】 (Conf: {min_conf:.2f}) | Attr: {COLOR_TEXT[pred_color_id]}")


def batch_test():
    model = load_model()
    if not model: return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 获取所有图片
    image_list = glob.glob(os.path.join(TEST_DIR, "*.jpg")) + glob.glob(os.path.join(TEST_DIR, "*.png"))

    print(f"🚀 开始测试 ({len(image_list)} 张图片)...")
    print("-" * 60)
    for img_path in image_list:
        predict_single_image(img_path, model, transform)
    print("-" * 60)


if __name__ == "__main__":
    batch_test()