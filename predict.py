import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import os
import numpy as np

from model import MultiTaskRiceNet
from attributes_config import ATTRIBUTE_GT, DISEASE_MAPPING

# ================= 配置区域 =================
MODEL_PATH = "rice_multitask_model.pth"
TEST_IMAGE_PATH = r"D:\Rice_Disease_Project\test\1.jpg"  # 换成你的猫

# 【核心参数1】逻辑匹配阈值 (0-3)
# 必须满足几个属性才算确诊？建议严格点设为 3，或者宽松点设为 2
LOGIC_THRESHOLD = 3

# 【核心参数2】能量拒识阈值 (关键！)
# 这是一个“信号强度”的门槛。
# 如果你的猫被识别了，尝试把这个数字调大 (比如 5.0 -> 8.0 -> 10.0)
# 如果正常的水稻被拒识了，把这个数字调小
ENERGY_THRESHOLD = 5.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SHAPE_TEXT = {0: "纺锤形", 1: "圆形/椭圆", 2: "波浪/不规则", 3: "全叶变色"}
COLOR_TEXT = {0: "灰白色", 1: "褐色", 2: "橙黄色"}
LOC_TEXT = {0: "随机分布", 1: "叶尖/叶缘"}


def load_model():
    model = MultiTaskRiceNet()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    return None


def predict_and_reason(image_path, model):
    if not os.path.exists(image_path):
        return

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        img_raw = Image.open(image_path).convert('RGB')
        img_tensor = transform(img_raw).unsqueeze(0).to(DEVICE)
    except:
        print("图片读取错误")
        return

    # === 模型推理 ===
    with torch.no_grad():
        # 拿到原始的 Logits (没有经过 Softmax 的原始分)
        logit_shape, logit_color, logit_loc = model(img_tensor)

        # 1. 【能量检测】计算响应强度
        # 我们取三个头中最大响应值的平均数作为“自信度”
        energy_shape = torch.max(logit_shape).item()
        energy_color = torch.max(logit_color).item()
        energy_loc = torch.max(logit_loc).item()

        avg_energy = (energy_shape + energy_color + energy_loc) / 3.0

        # 计算属性ID
        shape_id = torch.argmax(logit_shape, dim=1).item()
        color_id = torch.argmax(logit_color, dim=1).item()
        loc_id = torch.argmax(logit_loc, dim=1).item()

    print("\n" + "=" * 50)
    print(f"📸 图像: {os.path.basename(image_path)}")
    print(f"⚡ 信号响应强度 (Energy Score): {avg_energy:.4f}")
    print("-" * 50)

    # === 第一道防线：能量拒识 ===
    if avg_energy < ENERGY_THRESHOLD:
        print(f"🛑 [系统报警] 异常图像拒识！")
        print(f"   原因：图像特征响应强度过低 ({avg_energy:.2f} < {ENERGY_THRESHOLD})。")
        print(f"   推测：这不是水稻叶片，或者图像质量极差。")
        print("=" * 50 + "\n")
        return  # 直接结束，不往下走了

    # === 第二道防线：逻辑推理 ===
    print("👁️ [视觉感知]")
    print(f"  • 形状: {SHAPE_TEXT[shape_id]}")
    print(f"  • 颜色: {COLOR_TEXT[color_id]}")
    print(f"  • 位置: {LOC_TEXT[loc_id]}")
    print("-" * 50)
    print("🧠 [逻辑推理]")

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

    print(f"  >> 最佳逻辑匹配: {best_match} (匹配度: {max_score}/3)")

    if max_score >= LOGIC_THRESHOLD:
        print(f"✅ 最终诊断: 【{best_match}】")
    else:
        print(f"⚠️ 逻辑拒识：虽然看起来像水稻，但特征组合不符合病理学定义。")
        print(f"   检测到的组合 (形:{shape_id}, 色:{color_id}, 位:{loc_id}) 无效。")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    model = load_model()
    if model:
        # 在这里填你的猫的路径
        predict_and_reason(TEST_IMAGE_PATH, model)