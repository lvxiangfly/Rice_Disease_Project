import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm


def find_min_vegetation_ratio(image_folder, h_min, h_max):
    print(f"🚀 正在分析植被占比 (H范围: {h_min}-{h_max})...")

    extensions = ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg']
    img_files = []
    for ext in extensions:
        img_files.extend(glob(os.path.join(image_folder, '**', ext), recursive=True))
    img_files = sorted(list(set(img_files)))
    if not img_files:
        print("❌ 未找到图片")
        return

    ratios = []

    for img_path in tqdm(img_files):
        try:
            img = cv2.imread(img_path)
            if img is None: continue

            # 缩放加速
            h, w = img.shape[:2]
            if w > 640:
                scale = 640 / w
                img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                h, w = img.shape[:2]  # 更新尺寸

            # 转HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h_chan = hsv[:, :, 0]
            s_chan = hsv[:, :, 1]
            v_chan = hsv[:, :, 2]

            # 1. 植被像素掩码 (使用最佳 HSV 范围 + 基础过滤)
            # 注意：这里要用您上一轮算出来的最佳 H 范围
            mask_bio = (h_chan >= h_min) & (h_chan <= h_max) & (s_chan > 40) & (v_chan > 40)

            # 2. 计算占比
            bio_pixels = np.count_nonzero(mask_bio)
            total_pixels = h * w
            ratio = bio_pixels / total_pixels
            ratios.append(ratio)

        except Exception as e:
            pass

    if not ratios:
        print("计算失败")
        return

    # --- 统计分析 ---
    ratios = np.array(ratios)
    min_val = np.min(ratios)
    mean_val = np.mean(ratios)
    p1 = np.percentile(ratios, 1)  # 第1百分位，排除极端的异常噪点

    print("\n" + "=" * 40)
    print("      🌱 植被占比统计报告      ")
    print("=" * 40)
    print(f"图片总数: {len(ratios)}")
    print(f"平均占比 (Mean): {mean_val:.4f} ({mean_val * 100:.2f}%)")
    print(f"最小占比 (Min) : {min_val:.4f} ({min_val * 100:.2f}%)")
    print(f"1% 分位点 (P1) : {p1:.4f} ({p1 * 100:.2f}%)")
    print("-" * 40)

    # 推荐阈值：取最小值的 80% 作为安全边际，或者取 P1 的 80%
    suggested_tau = min_val * 0.8
    print(f"✅ 建议设定阈值 tau_env = {suggested_tau:.4f}")
    print("=" * 40)


# ==========================================
if __name__ == "__main__":
    folder = r"D:\Rice_Disease_Project\image_folder"  # 改这里

    # 👇【重要】填入上一轮代码算出来的最佳 Min 和 Max
    BEST_MIN = 12
    BEST_MAX = 65

    find_min_vegetation_ratio(folder, BEST_MIN, BEST_MAX)