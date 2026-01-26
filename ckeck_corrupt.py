import os
import glob
from PIL import Image
from tqdm import tqdm
from src_unet.config import Config # 引用你的配置

def check_integrity(folder_name):
    # 拼接路径
    dir_path = os.path.join(Config.DATA_ROOT, folder_name)
    # 获取所有 png 文件
    files = sorted(glob.glob(os.path.join(dir_path, "*.png")))
    
    print(f"🕵️‍♂️ 正在检查 {folder_name} 中的 {len(files)} 个文件...")
    
    corrupt_files = []
    
    for file_path in tqdm(files):
        try:
            # 打开并尝试加载图像数据 (这一步会触发 truncated 错误)
            with Image.open(file_path) as img:
                img.load() 
        except OSError:
            print(f"\n❌ 发现损坏文件: {file_path}")
            corrupt_files.append(file_path)
        except Exception as e:
            print(f"\n❌ 读取错误 ({e}): {file_path}")
            corrupt_files.append(file_path)
            
    return corrupt_files

if __name__ == "__main__":
    # 1. 检查 Images
    bad_images = check_integrity("images")
    
    # 2. 检查 Labels (如果 image 坏了，label 最好也检查一下)
    # bad_labels = check_integrity("images")
    
    print("\n" + "="*30)
    print("📋 检查报告")
    if not bad_images:
        print("✅ 所有文件均完好无损！(那可能是MONAI缓存问题)")
    else:
        print(f"🚨 共发现 {len(bad_images)} 个损坏文件。")
        print("建议手动删除这些文件，或从备份中重新上传。")