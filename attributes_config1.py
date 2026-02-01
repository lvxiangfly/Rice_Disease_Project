# attributes_config.py

# ================= 1. 定义属性类别 (用于定义网络输出层大小) =================
# 形状 (3类)
SHAPE_CLASSES = ['Spindle', 'Round_Oval', 'Irregular_Whole']
# 颜色 (3类)
COLOR_CLASSES = ['Grey_Brown', 'Yellow_Orange', 'White_Pale']
# 位置 (3类)
LOC_CLASSES = ['Random', 'Tip_Edge', 'Whole_Leaf']

# ================= 2. 定义病害与属性的映射关系 (核心逻辑) =================
# 格式: { '病害文件夹名': {'shape': 索引, 'color': 索引, 'loc': 索引} }

# 假设您的文件夹名是英文 (请根据实际情况修改键名)
# 0: Rice_Blast (稻瘟病) -> 纺锤形(0), 灰褐色(0), 随机分布(0)
# 1: Brown_Spot (褐斑病) -> 圆形(1), 灰褐色(0), 随机分布(0)
# 2: Bacterial_Blight (白叶枯) -> 不规则/条纹(2), 灰白色(2), 叶尖叶缘(1)
# 3: Tungro (东格鲁) -> 全叶(2), 黄橙色(1), 全叶(2)

DISEASE_MAPPING = {
    '0_Blast': {'shape': 0, 'color': 0, 'loc': 0},
    '1_BrownSpot': {'shape': 1, 'color': 0, 'loc': 0},
    '2_Blight': {'shape': 2, 'color': 2, 'loc': 1},
    '3_Tungro': {'shape': 2, 'color': 1, 'loc': 2}
}


# 辅助函数：根据病害的 Class Index 获取 属性 Label
def get_attr_labels_by_index(class_idx, class_names):
    """
    输入: class_idx (int), class_names (list of strings)
    输出: shape_label, color_label, loc_label
    """
    disease_name = class_names[class_idx]

    # 容错处理：如果映射表里没这个名字，默认都给0
    if disease_name not in DISEASE_MAPPING:
        print(f"⚠️ 警告: 未知类别 {disease_name}，属性默认设为0")
        return 0, 0, 0

    attrs = DISEASE_MAPPING[disease_name]
    return attrs['shape'], attrs['color'], attrs['loc']