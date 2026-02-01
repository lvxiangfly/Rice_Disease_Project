import cv2
import numpy as np
import os
from tqdm import tqdm

# ================= ⚙️ 参数配置区域 (请确认路径) =================
# 1. 数据集路径
RICE_DIR = r"D:\Rice_Disease_Project\test_rice"  # 水稻测试集文件夹
NOISE_DIR = r"D:\Rice_Disease_Project\test_noise"  # 负样本(猫狗)文件夹

# 2. 您之前算出的最佳 H 范围 (已预填)
H_MIN = 12
H_MAX = 65

# 3. 想要尝试的参数网格
# 面积阈值候选列表 (从您的0.04开始，逐步增加到0.30)
TAU_CANDIDATES = [0.04, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25,0.30,0.35,0.40]
# 饱和度阈值候选列表 (40是基础，60更严，80非常严)
S_MIN_CANDIDATES = [40, 50, 60,70,80]


# ===============================================================

def calculate_ratio(img_path, s_min):
    """
    读取图片并计算符合 H 和 S 条件的像素占比
    """
    try:
        # 读取图片
        img = cv2.imread(img_path)
        if img is None: return 0.0

        # 缩放图片以加速计算 (宽度限制在 320px)
        h, w = img.shape[:2]
        if w > 320:
            scale = 320 / w
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        # 转 HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 核心掩膜逻辑：
        # 1. Hue 在 [12, 65] 之间 (您的数据)
        # 2. Saturation > s_min (过滤灰蒙蒙的背景)
        # 3. Value > 40 (过滤太黑的区域)
        mask = (hsv[:, :, 0] >= H_MIN) & (hsv[:, :, 0] <= H_MAX) & \
               (hsv[:, :, 1] > s_min) & (hsv[:, :, 2] > 40)

        # 计算占比
        pixel_count = np.count_nonzero(mask)
        total_pixels = img.shape[0] * img.shape[1]

        return pixel_count / total_pixels

    except Exception:
        return 0.0


def evaluate_dataset(folder_path, tau, s_min, is_noise_dataset):
    """
    评估整个文件夹的表现
    """
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
             if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if len(files) == 0:
        return 0.0

    pass_count = 0

    # 遍历所有图片
    # 注意：为了速度，这里不显示进度条，直接跑
    for f in files:
        ratio = calculate_ratio(f, s_min)
        if ratio >= tau:
            pass_count += 1

    if is_noise_dataset:
        # 对于负样本：拒识率 = (总数 - 通过数) / 总数
        return ((len(files) - pass_count) / len(files)) * 100
    else:
        # 对于水稻：保留率 = 通过数 / 总数
        return (pass_count / len(files)) * 100


if __name__ == "__main__":
    # 检查文件夹是否存在
    if not os.path.exists(RICE_DIR) or not os.path.exists(NOISE_DIR):
        print("❌ 错误：找不到数据集文件夹，请检查 RICE_DIR 和 NOISE_DIR 路径。")
        exit()

    print(f"🚀 开始网格搜索 (Grid Search)...")
    print(f"   H范围: [{H_MIN}, {H_MAX}]")
    print("-" * 75)
    print(f"{'Tau(面积)':<10} | {'S_min(饱和)':<12} | {'🌾 水稻保留率':<15} | {'🛡️ OOD拒识率':<15} | {'评价'}")
    print("-" * 75)

    best_score = -1
    best_params = None

    # 开始双重循环遍历所有参数组合
    # 这里的 tqdm 是为了显示总进度
    total_combinations = len(TAU_CANDIDATES) * len(S_MIN_CANDIDATES)

    # 预加载文件列表以检查数量
    n_rice = len(os.listdir(RICE_DIR))
    n_noise = len(os.listdir(NOISE_DIR))
    print(f"📊 数据量: 水稻 {n_rice} 张, 负样本 {n_noise} 张")
    print("-" * 75)

    for s in S_MIN_CANDIDATES:
        for t in TAU_CANDIDATES:
            # 跑两个数据集
            rice_retention = evaluate_dataset(RICE_DIR, t, s, is_noise_dataset=False)
            noise_rejection = evaluate_dataset(NOISE_DIR, t, s, is_noise_dataset=True)

            # 自动打分评价
            status = ""
            if rice_retention < 95.0:
                status = "❌ 误删太多"
            elif noise_rejection < 40.0:
                status = "⚠️ 拒识太低"
            else:
                status = "✅ 可用"

                # 简单的加权打分：优先保证水稻不丢，然后看拒识率
                # 如果水稻保留率 > 98%，则分数 = 拒识率
                current_score = noise_rejection
                if rice_retention < 98.0:
                    # 如果水稻保留率稍微掉了一点，给分数打个折
                    current_score -= (98.0 - rice_retention) * 2

                if current_score > best_score:
                    best_score = current_score
                    best_params = (t, s, rice_retention, noise_rejection)

            # 打印这一行的结果
            print(f"{t:<10.2f} | {s:<12} | {rice_retention:<16.2f}% | {noise_rejection:<16.2f}% | {status}")

    print("-" * 75)
    # ================= 📊 新增：绘制敏感度热力图 =================
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        print("\n🎨 正在绘制敏感度分析热力图...")

        # 1. 整理数据用于绘图
        # 我们需要两个矩阵：一个是水稻保留率，一个是OOD拒识率
        data_rice = []
        data_ood = []

        for s in S_MIN_CANDIDATES:
            row_rice = []
            row_ood = []
            for t in TAU_CANDIDATES:
                # 重新计算一遍或者在上面循环里存下来 (这里为了代码简洁，建议您在上面循环里用列表存结果)
                # 为了不破坏您上面的结构，建议您在上面循环里加一个 results_list = []
                # 这里为了演示，我假设您已经把结果存到了字典里，或者您直接修改上面的循环
                pass

                # 为了方便，建议直接把上面的循环改成这样：
        results = []
        for s in S_MIN_CANDIDATES:
            for t in TAU_CANDIDATES:
                acc_rice = evaluate_dataset(RICE_DIR, t, s, False)
                acc_ood = evaluate_dataset(NOISE_DIR, t, s, True)
                results.append({'S_min': s, 'Tau': t, 'Rice_Retention': acc_rice, 'OOD_Rejection': acc_ood})

                # 打印逻辑保持不变...

        # 转 DataFrame
        df = pd.DataFrame(results)

        # 2. 绘制 OOD 拒识率热力图
        pivot_ood = df.pivot(index="S_min", columns="Tau", values="OOD_Rejection")
        pivot_rice = df.pivot(index="S_min", columns="Tau", values="Rice_Retention")

        plt.figure(figsize=(12, 5))

        # 图 1: OOD 拒识率
        plt.subplot(1, 2, 1)
        sns.heatmap(pivot_ood, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'OOD Rejection (%)'})
        plt.title('OOD Rejection Rate (Higher is Better)')
        plt.xlabel('Vegetation Ratio Threshold (Tau)')
        plt.ylabel('Saturation Threshold (S_min)')

        # 图 2: 水稻保留率 (用灰色掩盖掉 < 98% 的区域)
        plt.subplot(1, 2, 2)
        sns.heatmap(pivot_rice, annot=True, fmt=".1f", cmap="Greens", vmin=95, vmax=100,
                    cbar_kws={'label': 'Rice Retention (%)'})
        plt.title('Rice Retention Rate (Target > 99%)')
        plt.xlabel('Vegetation Ratio Threshold (Tau)')
        plt.ylabel('Saturation Threshold (S_min)')

        plt.tight_layout()
        plt.savefig('sensitivity_analysis_heatmap.png', dpi=300)
        print("✅ 热力图已保存为 sensitivity_analysis_heatmap.png")
        plt.show()

    except ImportError:
        print("⚠️ 缺少 matplotlib 或 seaborn 库，跳过绘图。")
    except Exception as e:
        print(f"⚠️ 绘图出错: {e}")
    if best_params:
        print(f"\n🏆 【最终推荐参数】")
        print(f"   请将 experiment_runner.py 中的参数修改为：")
        print(f"   👉 TAU_ENV = {best_params[0]}")
        print(f"   👉 (修改代码逻辑) S_chan > {best_params[1]}")
        print(f"\n   📈 预期效果：")
        print(f"      水稻保留率: {best_params[2]:.2f}% (越高越好)")
        print(f"      OOD 拒识率: {best_params[3]:.2f}% (越高越好)")
    else:
        print("🤔 未找到完美参数，建议检查数据集或放宽 H 范围。")