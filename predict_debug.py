import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import os
import cv2
import numpy as np
import glob

from model import MultiTaskRiceNet
from attributes_config import ATTRIBUTE_GT, DISEASE_MAPPING

# ================= ⚙️ 核心配置 =================
MODEL_PATH = "rice_multitask_model.pth"
TEST_DIR = r"D:\Rice_Disease_Project\test_images"

# 1. 概率阈值 (0.70)
CONFIDENCE_THRESHOLD = 0.70

# 2. 绿色生机阈值 (7%)
MIN_GREEN_RATIO = 0.07

# 3. 病斑真实性阈值 (0.5%)
MIN_DISEASE_RATIO = 0.005

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 属性文字
SHAPE_TEXT = {0: "纺锤形", 1: "圆形/椭圆", 2: "波浪/不规则", 3: "全叶变色"}
COLOR_TEXT = {0: "灰白色", 1: "褐色", 2: "橙黄色"}
LOC_TEXT = {0: "随机分布", 1: "叶尖/叶缘"}


# ================= 🛡️ 辅助函数 =================

def detect_face(img_bgr):
    """检测人脸"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty(): return False

    faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(30, 30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            roi = img_bgr[y:y + h, x:x + w]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask_green = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([90, 255, 255]))
            if np.count_nonzero(mask_green) / (w * h) < 0.4: return True
    return False


def is_valid_plant_environment(img_bgr):
    """检测是否是植物环境"""
    h, w = img_bgr.shape[:2]
    crop_h, crop_w = int(h * 0.2), int(w * 0.2)
    center_img = img_bgr[crop_h:h - crop_h, crop_w:w - crop_w]

    hsv = cv2.cvtColor(center_img, cv2.COLOR_BGR2HSV)
    total = center_img.shape[0] * center_img.shape[1]

    mask_green = cv2.inRange(hsv, np.array([35, 30, 30]), np.array([85, 255, 255]))
    green_ratio = np.count_nonzero(mask_green) / total

    mask_brown = cv2.inRange(hsv, np.array([10, 40, 40]), np.array([35, 255, 255]))
    mask_gray = cv2.inRange(hsv, np.array([0, 5, 50]), np.array([180, 50, 255]))

    plant_mask = cv2.bitwise_or(mask_green, mask_brown)
    plant_mask = cv2.bitwise_or(plant_mask, mask_gray)
    plant_ratio = np.count_nonzero(plant_mask) / total

    if green_ratio < MIN_GREEN_RATIO:
        return False, green_ratio, f"缺乏绿色生机 ({green_ratio:.1%})"

    if plant_ratio < 0.15:
        return False, green_ratio, f"非植物环境 (植物色{plant_ratio:.1%})"

    return True, green_ratio, "通过"


def verify_disease_color(img_bgr, pred_color_id, green_ratio):
    """验伤环节 (杀花盆)"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    total = img_bgr.shape[0] * img_bgr.shape[1]

    target_ratio = 0.0
    color_name = COLOR_TEXT[pred_color_id]

    if pred_color_id == 0:  # 灰白色
        # 排除纯灰花盆 (S<5)
        mask_warm = cv2.inRange(hsv, np.array([10, 5, 50]), np.array([40, 60, 255]))
        mask_cool = cv2.inRange(hsv, np.array([0, 8, 50]), np.array([180, 50, 220]))
        mask_final = cv2.bitwise_or(mask_warm, mask_cool)
        target_ratio = np.count_nonzero(mask_final) / total

    elif pred_color_id == 1:  # 褐色
        mask = cv2.inRange(hsv, np.array([10, 30, 20]), np.array([30, 255, 200]))
        target_ratio = np.count_nonzero(mask) / total

    elif pred_color_id == 2:  # 橙黄色
        mask = cv2.inRange(hsv, np.array([15, 40, 50]), np.array([40, 255, 255]))
        target_ratio = np.count_nonzero(mask) / total

    if green_ratio > 0.20 and target_ratio < MIN_DISEASE_RATIO:
        return False, f"健康绿植误报 (有效{color_name}仅占{target_ratio:.1%})"

    return True, f"有效病斑占比 {target_ratio:.1%}"


# ================= 🚀 主流程 (增强调试版) =================

def load_model():
    print("📥 正在加载模型...")
    model = MultiTaskRiceNet()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("✅ 模型加载成功")
        return model
    print(f"❌ 模型文件不存在: {MODEL_PATH}")
    return None


def predict_single_image(image_path, model, transform):
    filename = os.path.basename(image_path)
    print(f"\n" + "=" * 60)
    print(f"🖼️  正在分析图片: {filename}")

    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print("❌ 读取图片失败")
        return

    # --- Step 1: 人脸检测 ---
    print("1️⃣  [Step 1] 人脸安全检测...", end=" ")
    if detect_face(img_cv):
        print("❌ 拦截! (检测到人脸特征)")
        return
    print("✅ 通过")

    # --- Step 2: 环境检测 ---
    print("2️⃣  [Step 2] 农田环境检测...", end=" ")
    is_plant, green_ratio, msg = is_valid_plant_environment(img_cv)
    if not is_plant:
        print(f"❌ 拦截! ({msg})")
        return
    print(f"✅ 通过 (绿色占比: {green_ratio:.1%})")

    # 预处理
    try:
        img_pil = Image.open(image_path).convert('RGB')
        img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    except:
        print("❌ 图片格式转换错误")
        return

    # --- Step 3: 模型推理 (重点修改) ---
    print("\n3️⃣  [Step 3] 🧠 三头神经网络判决详情:")
    with torch.no_grad():
        logit_shape, logit_color, logit_loc = model(img_tensor)

        # 计算概率
        prob_shape = F.softmax(logit_shape, dim=1)
        prob_color = F.softmax(logit_color, dim=1)
        prob_loc = F.softmax(logit_loc, dim=1)

        # 获取最大值
        conf_shape, shape_id = torch.max(prob_shape, 1)
        conf_color, color_id = torch.max(prob_color, 1)
        conf_loc, loc_id = torch.max(prob_loc, 1)

        # 转为 Python数值
        s_id, c_id, l_id = shape_id.item(), color_id.item(), loc_id.item()
        s_conf, c_conf, l_conf = conf_shape.item(), conf_color.item(), conf_loc.item()

        min_conf = min(s_conf, c_conf, l_conf)

        # 打印三头详情
        print(f"    🔸 [形状头] 预测: {SHAPE_TEXT[s_id]:<8} | 置信度: {s_conf:.2%}")
        print(f"    🔸 [颜色头] 预测: {COLOR_TEXT[c_id]:<8} | 置信度: {c_conf:.2%}")
        print(f"    🔸 [位置头] 预测: {LOC_TEXT[l_id]:<8} | 置信度: {l_conf:.2%}")
        print(f"    ------------------------------------------")
        print(f"    ⚖️  木桶短板(最低置信度): {min_conf:.2%}")

    # --- Step 4: 置信度过滤 ---
    if min_conf < CONFIDENCE_THRESHOLD:
        print(f"❌ [Step 4] 拦截! 模型太犹豫 (阈值 {CONFIDENCE_THRESHOLD:.0%}) -> 判定为未知/干扰物体")
        return
    else:
        print(f"✅ [Step 4] 通过! 模型由于置信度足够高")

    # --- Step 5: 验伤 ---
    print(f"5️⃣  [Step 5] 视觉一致性校验 (针对{COLOR_TEXT[c_id]})...", end=" ")
    is_sick, sick_msg = verify_disease_color(img_cv, c_id, green_ratio)
    if not is_sick:
        print(f"❌ 拦截! ({sick_msg})")
        print("    -> 判定为: 健康绿植 / 花盆误报")
        return
    print(f"✅ 通过 ({sick_msg})")

    # --- Step 6: 逻辑组合 ---
    print("6️⃣  [Step 6] 病理逻辑匹配...", end=" ")
    max_score = 0
    best_match = "Unknown"
    for disease_id, rules in ATTRIBUTE_GT.items():
        score = 0
        if rules['shape'] == s_id: score += 1
        if rules['color'] == c_id: score += 1
        if rules['loc'] == l_id: score += 1
        if score > max_score:
            max_score = score
            best_match = DISEASE_MAPPING[disease_id]

    if max_score < 2:
        print(f"❌ 失败 (逻辑得分 {max_score}/3)")
        print(f"    -> 当前组合 [{SHAPE_TEXT[s_id]} + {COLOR_TEXT[c_id]} + {LOC_TEXT[l_id]}] 不符合任何已知病害")
        return

    print(f"✅ 匹配成功 ({best_match})")

    print(f"\n🎉 最终诊断: 【{best_match}】")
    print("=" * 60)


def batch_test():
    model = load_model()
    if not model: return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_list = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_list.extend(glob.glob(os.path.join(TEST_DIR, ext)))

    if not image_list:
        print("没有找到图片，请检查路径")
        return

    for img_path in image_list:
        predict_single_image(img_path, model, transform)


if __name__ == "__main__":
    batch_test()