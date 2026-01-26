import os
import glob
import random
from PIL import Image
from tqdm import tqdm

# 配置你的路径
data_root = "Data/BUS_Mix_Seg" # 你的路径

def check_raw_files(folder, sample_num=100):
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    print(f"📂 正在检查 {folder} (共 {len(files)} 张)...")
    
    # 随机抽查
    samples = random.sample(files, min(len(files), sample_num))
    
    modes = {}
    shapes = []
    
    for f in samples:
        with Image.open(f) as img:
            # 统计模式 (L=灰度, RGB=彩色, RGBA=带透明度)
            modes[img.mode] = modes.get(img.mode, 0) + 1
            shapes.append(img.size) # (W, H)
            
    print(f"   👉 模式分布: {modes}")
    print(f"   👉 尺寸示例 (W, H): {shapes[:3]}")
    return modes

print("=== 1. 检查原始文件 ===")
image_modes = check_raw_files(os.path.join(data_root, "images"))
label_modes = check_raw_files(os.path.join(data_root, "labels"))

# 诊断逻辑
if "RGB" in image_modes and "L" in image_modes:
    print("\n❌ 警告：Images 文件夹里混杂了 RGB 和 灰度图！这会导致 Tensor 维度不一致。")
elif "RGB" in image_modes:
    print("\nℹ️ 确认：Images 全是 RGB 格式。")

if "RGB" in label_modes:
    print("\n❌ 警告：Labels 竟然是 RGB 图！请确认标签应为灰度图(L)格式。")
if "L" not in label_modes:
    print("\n❌ 警告：Labels 不是灰度图(L)，可能是 RGB？如果是 RGB，MONAI 会把它当成 3 通道读取。")