import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from src_unet.config import Config

def check_label_shapes():
    # 你的数据根目录
    root_dir = Config.DATA_ROOT
    
    # 获取所有 label 文件
    label_pattern = os.path.join(root_dir, "labels", "*.png")
    labels = glob.glob(label_pattern)
    
    print(f"🕵️‍♂️ 正在检查 {len(labels)} 个 Label 文件的形状...")
    
    error_count = 0
    
    for path in tqdm(labels):
        try:
            # 使用 PIL 读取
            with Image.open(path) as img:
                # 转为 numpy 查看形状
                arr = np.array(img)
                
                # 检查是否为 3 维 (即 H, W, 3)
                # 正常的 Mask 应该是 (H, W) -> ndim=2
                if arr.ndim == 3:
                    print(f"\n⚠️ 发现 RGB 格式 Mask (3通道):")
                    print(f"   - 文件: {os.path.basename(path)}")
                    print(f"   - 形状: {arr.shape}")
                    print(f"   - 来源: {os.path.basename(path).split('_')[0]}")
                    error_count += 1
                    
                    # 只打印前5个，避免刷屏
                    if error_count >= 5:
                        print("... 还有更多，请修复逻辑 ...")
                        return

        except Exception as e:
            print(f"❌ 读取错误: {path} | {e}")

    if error_count == 0:
        print("✅ 所有 Label 看起来都是正常的灰度图 (2D)。可能是 Image 的问题？")
    else:
        print(f"\n🚨 总计发现至少 {error_count} 个 RGB Mask，需要修改 dataset.py 里的处理逻辑。")

if __name__ == "__main__":
    check_label_shapes()