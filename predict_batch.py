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
# 只有模型非常确信时才放行，这能有效拦截人脸和模糊物体
CONFIDENCE_THRESHOLD = 0.70

# 2. 绿色生机阈值 (7%)
# 【杀猫利器】全图必须有 7% 以上是明显的绿色，否则视为非农作物
MIN_GREEN_RATIO = 0.07

# 3. 病斑真实性阈值 (0.5%)
# 【杀花盆利器】如果模型说是病，但图里找不到"生物病斑色"，视为健康绿植
MIN_DISEASE_RATIO = 0.005

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 属性文字
SHAPE_TEXT = {0: "纺锤形", 1: "圆形/椭圆", 2: "波浪/不规则", 3: "全叶变色"}
COLOR_TEXT = {0: "灰白色", 1: "褐色", 2: "橙黄色"}
LOC_TEXT = {0: "随机分布", 1: "叶尖/叶缘"}


# ================= 🛡️ 功能 1：环境与人脸检测 =================

def detect_face(img_bgr):
    """强制人脸检测"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty(): return False

    # 稍微严格的参数，防止把叶脉纹理当人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(30, 30))

    if len(faces) > 0:
        # 二次确认：如果"脸"全是绿色的，那是叶子
        for (x, y, w, h) in faces:
            face_roi = img_bgr[y:y + h, x:x + w]
            hsv_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            # 宽泛绿色
            mask_green = cv2.inRange(hsv_roi, np.array([30, 30, 30]), np.array([90, 255, 255]))
            green_ratio = np.count_nonzero(mask_green) / (w * h)

            if green_ratio < 0.4: return True  # 绿色少，是真人脸
    return False


def is_valid_plant_environment(img_bgr):
    """
    检查环境：
    1. 必须有绿色 (杀猫)
    2. 必须有植物色 (杀杂物)
    """
    h, w = img_bgr.shape[:2]
    # 取中心 60% 区域，避开边缘背景干扰
    crop_h, crop_w = int(h * 0.2), int(w * 0.2)
    center_img = img_bgr[crop_h:h - crop_h, crop_w:w - crop_w]

    hsv = cv2.cvtColor(center_img, cv2.COLOR_BGR2HSV)
    total = center_img.shape[0] * center_img.shape[1]

    # === A. 绿色检测 (严格版) ===
    # H: 35-85 (踢掉偏黄的杂草色), S: >30 (踢掉发白的背景)
    mask_green = cv2.inRange(hsv, np.array([35, 30, 30]), np.array([85, 255, 255]))
    green_ratio = np.count_nonzero(mask_green) / total

    # === B. 泛植物色检测 ===
    # 褐/黄 (枯叶)
    mask_brown = cv2.inRange(hsv, np.array([10, 40, 40]), np.array([35, 255, 255]))
    # 灰 (病斑)
    mask_gray = cv2.inRange(hsv, np.array([0, 5, 50]), np.array([180, 50, 255]))

    total_plant_pixels = cv2.bitwise_or(mask_green, mask_brown)
    total_plant_pixels = cv2.bitwise_or(total_plant_pixels, mask_gray)
    plant_ratio = np.count_nonzero(total_plant_pixels) / total

    # 规则 1：如果没有绿色，判定为非作物 (橘猫在此被杀)
    if green_ratio < MIN_GREEN_RATIO:
        return False, green_ratio, f"缺乏绿色生机 ({green_ratio:.1%})"

    # 规则 2：如果连一点植物色都没有，判定为杂物
    if plant_ratio < 0.15:
        return False, green_ratio, f"非植物环境 (植物色{plant_ratio:.1%})"

    return True, green_ratio, "通过"


# ================= 🛡️ 功能 2：验伤 (生物灰滤镜) =================

def verify_disease_color(img_bgr, pred_color_id, green_ratio):
    """
    【核心修改】
    区分"生物灰"(病斑)和"工业灰"(花盆)。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    total = img_bgr.shape[0] * img_bgr.shape[1]

    target_ratio = 0.0
    color_name = COLOR_TEXT[pred_color_id]

    if pred_color_id == 0:  # 灰白色 (White/Gray)
        # 【杀花盆逻辑】
        # 花盆通常是: S < 5 (极低饱和度), V > 50 (亮灰/白)
        # 病斑通常是: S > 5 (带点枯黄), 或者 H 在黄色区间

        # 定义"生物灰" (排除纯色花盆):
        # 1. 暖灰色: H: 10-40 (微黄), S: 5-60
        mask_warm_gray = cv2.inRange(hsv, np.array([10, 5, 50]), np.array([40, 60, 255]))

        # 2. 冷灰色: H: 0-180, S: 8-50 (必须有一定饱和度，不能是纯灰)
        mask_cool_gray = cv2.inRange(hsv, np.array([0, 8, 50]), np.array([180, 50, 220]))

        mask_final = cv2.bitwise_or(mask_warm_gray, mask_cool_gray)
        target_ratio = np.count_nonzero(mask_final) / total

    elif pred_color_id == 1:  # 褐色 (Brown)
        # H: 10-30
        mask = cv2.inRange(hsv, np.array([10, 30, 20]), np.array([30, 255, 200]))
        target_ratio = np.count_nonzero(mask) / total

    elif pred_color_id == 2:  # 橙黄色 (Orange)
        # H: 15-40
        mask = cv2.inRange(hsv, np.array([15, 40, 50]), np.array([40, 255, 255]))
        target_ratio = np.count_nonzero(mask) / total

    # === 判决 ===
    # 如果图片是绿的 (>20%)，但找不到符合定义的"生物病斑" (<0.5%)
    # 那么即使有大面积的灰色花盆(S<5)，这里计算出的 target_ratio 也会是 0
    if green_ratio > 0.20 and target_ratio < MIN_DISEASE_RATIO:
        return False, f"健康绿植误报 (有效{color_name}仅占{target_ratio:.1%})"

    return True, f"病斑占比{target_ratio:.1%}"


# ================= 🚀 主流程 =================

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
    if img_cv is None: return {"file": filename, "status": "ERROR", "reason": "读取失败"}

    # 1. 人脸检测 (OpenCV)
    if detect_face(img_cv):
        return {"file": filename, "status": "REJECT_FACE", "reason": "检测到人脸特征"}

    # 2. 环境过滤 (杀猫)
    is_plant, green_ratio, msg = is_valid_plant_environment(img_cv)
    if not is_plant:
        return {"file": filename, "status": "REJECT_ENV", "reason": msg}

    # 预处理
    try:
        img_pil = Image.open(image_path).convert('RGB')
        img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    except:
        return {"file": filename, "status": "ERROR", "reason": "文件损坏"}

    # --- 模型推理 ---
    with torch.no_grad():
        logit_shape, logit_color, logit_loc = model(img_tensor)

        # 获取置信度
        conf_shape, shape_id = torch.max(F.softmax(logit_shape, dim=1), 1)
        conf_color, color_id = torch.max(F.softmax(logit_color, dim=1), 1)
        conf_loc, loc_id = torch.max(F.softmax(logit_loc, dim=1), 1)

        shape_id, color_id, loc_id = shape_id.item(), color_id.item(), loc_id.item()
        min_conf = min(conf_shape.item(), conf_color.item(), conf_loc.item())

    # 3. 置信度过滤
    if min_conf < CONFIDENCE_THRESHOLD:
        return {
            "file": filename,
            "status": "REJECT_CONF",
            "reason": f"模型不确定 (信度 {min_conf:.2f})"
        }

    # 4. 验伤 (杀绿植花盆)
    # 这一步会忽略 S<8 的纯灰色花盆，导致 target_ratio 极低，从而触发健康误报判定
    is_sick, sick_msg = verify_disease_color(img_cv, color_id, green_ratio)
    if not is_sick:
        return {"file": filename, "status": "REJECT_HEALTHY", "reason": sick_msg}

    # 5. 逻辑组合
    max_score = 0
    best_match = "Unknown"
    for disease_id, rules in ATTRIBUTE_GT.items():
        score = 0
        if rules['shape'] == shape_id: score += 1
        if rules['color'] == color_id: score += 1
        if rules['loc'] == loc_id:   score += 1
        if score > max_score:
            max_score = score
            best_match = DISEASE_MAPPING[disease_id]

    attr_desc = f"[{SHAPE_TEXT[shape_id]}/{COLOR_TEXT[color_id]}/{LOC_TEXT[loc_id]}]"

    if max_score < 2:
        return {"file": filename, "status": "REJECT_LOGIC", "reason": f"属性组合混乱 {attr_desc}"}

    return {
        "file": filename,
        "status": "SUCCESS",
        "diagnosis": best_match,
        "attributes": attr_desc,
        "conf": min_conf
    }


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

    print(f"\n🚀 批量测试 (Final Pro+)")
    print(f"🔧 策略: 生物灰滤镜(防花盆) | 绿色>7%(防猫) | 人脸检测")
    print("=" * 90)
    print(f"{'文件名':<20} | {'状态':<15} | {'结果/原因':<45}")
    print("-" * 90)

    for img_path in image_list:
        res = predict_single_image(img_path, model, transform)
        fname = res['file']
        if len(fname) > 18: fname = fname[:15] + "..."

        if res['status'] == "SUCCESS":
            print(f"{fname:<20} | ✅ 确诊         | 【{res['diagnosis']}】 (信度:{res['conf']:.2f})")
        elif "REJECT" in res['status']:
            print(f"{fname:<20} | 🛑 {res['status']} | {res['reason']}")
        else:
            print(f"{fname:<20} | ⚠️ 错误         | {res['reason']}")

    print("=" * 90)


if __name__ == "__main__":
    batch_test()