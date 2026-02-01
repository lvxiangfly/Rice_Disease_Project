import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

# 尝试导入进度条库，如果没有安装也不会报错
try:
    from tqdm import tqdm

    USE_TQDM = True
except ImportError:
    USE_TQDM = False


def calculate_optimal_thresholds(image_folder, output_plot='optimized_hue_analysis.png'):
    print(f"🚀 开始分析文件夹: {image_folder}")

    # --- 1. 获取所有图片 ---
    extensions = ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg']
    img_files = []
    for ext in extensions:
        img_files.extend(glob(os.path.join(image_folder, '**', ext), recursive=True))
    img_files = sorted(list(set(img_files)))
    total_files = len(img_files)
    if total_files == 0:
        print("❌ 错误：未找到图片，请检查路径是否正确。")
        return

    print(f"📄 共发现 {total_files} 张图片，正在提取像素特征...")

    # --- 2. 遍历图片并统计直方图 ---
    # 初始化 HSV 直方图 (H范围 0-180)
    total_hist = np.zeros(180, dtype=np.float64)
    valid_pixel_count = 0

    # 根据是否安装了 tqdm 选择迭代器
    iterator = tqdm(img_files, desc="Processing") if USE_TQDM else img_files

    for idx, img_path in enumerate(iterator):
        if not USE_TQDM and idx % 50 == 0:
            print(f"   正在处理: {idx}/{total_files}...")

        try:
            img = cv2.imread(img_path)
            if img is None: continue

            # 缩放图片以加速处理 (宽度限制在 640px)
            h, w = img.shape[:2]
            if w > 640:
                scale = 640 / w
                img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

            # 转换到 HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h_chan = hsv[:, :, 0]
            s_chan = hsv[:, :, 1]
            v_chan = hsv[:, :, 2]

            # === 关键预处理 ===
            # 过滤掉背景：饱和度(S)过低是灰色/白色，亮度(V)过低是黑色
            # 这里的 40 是经验值，用于排除花盆、泥土和背景板
            mask = (s_chan > 40) & (v_chan > 40)

            valid_h = h_chan[mask]

            if len(valid_h) > 0:
                # 累加直方图
                hist, _ = np.histogram(valid_h, bins=180, range=(0, 180))
                total_hist += hist
                valid_pixel_count += len(valid_h)

        except Exception as e:
            print(f"⚠️ 跳过坏图 {os.path.basename(img_path)}: {e}")

    # --- 3. 核心算法：基于CDF计算最佳阈值 ---
    print("\n🧮 正在计算最佳统计阈值 (Target Coverage: 95%)...")

    # 归一化直方图
    if valid_pixel_count == 0:
        print("❌ 错误：没有检测到任何有效绿色像素。")
        return

    prob_dist = total_hist / valid_pixel_count
    cdf = np.cumsum(prob_dist)  # 累积分布函数

    # 寻找包含 95% 数据的中心区间 (即剔除两端各 2.5% 的极端值)
    # searchsorted 返回满足条件的第一个索引
    optimal_min = np.searchsorted(cdf, 0.025)
    optimal_max = np.searchsorted(cdf, 0.975)

    # 修正：如果最大值太接近180，可能是有红色溢出，稍微限制一下
    if optimal_max > 170: optimal_max = 170

    # 计算覆盖率
    def get_coverage(low, high, hist):
        count = np.sum(hist[low: high + 1])
        return (count / np.sum(hist)) * 100

    old_cov = get_coverage(35, 90, total_hist)
    new_cov = get_coverage(optimal_min, optimal_max, total_hist)

    # --- 4. 打印最终报告 ---
    print("\n" + "=" * 40)
    print("      ✅ 分析完成 (Analysis Report)      ")
    print("=" * 40)
    print(f"原始预设范围 [35, 90] 覆盖率 : {old_cov:.2f}%")
    print("-" * 40)
    print(f"🌟 推荐最佳范围 (覆盖95%数据) : [{optimal_min}, {optimal_max}]")
    print(f"🌟 优化后覆盖率             : {new_cov:.2f}%")
    print("=" * 40)

    # --- 5. 绘制专业论文图表 ---
    plt.figure(figsize=(10, 6), dpi=150)

    x = np.arange(180)
    # 绘制背景分布
    plt.bar(x, total_hist, color='#e0e0e0', width=1.0, label='Pixel Distribution')

    # 绘制新范围 (绿色半透明区域)
    plt.bar(x[optimal_min:optimal_max + 1], total_hist[optimal_min:optimal_max + 1],
            color='#2ca02c', alpha=0.8, width=1.0,
            label=f'Optimized Range [{optimal_min}, {optimal_max}] (95%)')

    # 绘制旧范围 (红色虚线框)
    plt.axvline(35, color='red', linestyle='--', alpha=0.6, linewidth=1)
    plt.axvline(90, color='red', linestyle='--', alpha=0.6, linewidth=1)
    # 在旧范围上方加个文字标注
    plt.text(62.5, np.max(total_hist) * 0.85, "Original [35,90]",
             color='red', ha='center', fontsize=9, alpha=0.7)

    # 装饰
    plt.title(f'Hue Distribution & Threshold Optimization (N={total_files})', fontsize=14)
    plt.xlabel('Hue Value (OpenCV Scale: 0-180)', fontsize=12)
    plt.ylabel('Pixel Frequency', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle=':', alpha=0.5)

    # 添加结果文本框
    result_text = (f"Original Cov: {old_cov:.1f}%\n"
                   f"Optimized Cov: {new_cov:.1f}%\n"
                   f"Best Range: H $\\in$ [{optimal_min}, {optimal_max}]")
    plt.text(0, np.max(total_hist) * 0.95, result_text,
             fontsize=10, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"\n📊 统计图表已保存至: {os.path.abspath(output_plot)}")
    plt.show()


# ==========================================
# 👇 请在这里修改您的文件夹路径
# ==========================================
if __name__ == "__main__":
    # 替换为您存放图片的实际路径
    # 注意：如果路径里有中文，请确保 Python 编码设置正确
    folder_path = r"D:\Rice_Disease_Project\image_folder"

    if os.path.exists(folder_path):
        calculate_optimal_thresholds(folder_path)
    else:
        print(f"❌ 路径不存在: {folder_path}")