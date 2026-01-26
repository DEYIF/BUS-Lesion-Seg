import os
import glob
from PIL import Image
from tqdm import tqdm
import numpy as np

# === 配置路径 ===
# 请修改为你的实际路径
DATA_ROOT = "/home/maia-user/yufeng/Data/BUS_Mix_Cls" 

def convert_to_png(folder_path, is_label=False):
    """
    遍历文件夹，将所有非PNG图片转换为PNG，并统一格式。
    is_label: 如果是标签，需要做二值化处理，消除JPG噪点。
    """
    # 查找所有文件 (包括 png, jpg, jpeg, bmp 等)
    # 使用 set 去重，防止 glob 重复抓取
    files = set(glob.glob(os.path.join(folder_path, "*.*")))
    
    print(f"正在处理 {folder_path} ... 共找到 {len(files)} 个文件")
    
    # 进度条
    for filepath in tqdm(files):
        ext = os.path.splitext(filepath)[1].lower()
        
        # 即使原本是 png，如果是 Label，也建议重新跑一遍二值化，确保干净
        if ext not in ['.jpg', '.jpeg', '.bmp', '.png']:
            continue
            
        try:
            with Image.open(filepath) as img:
                # === 1. 处理 Label (关键步骤) ===
                if is_label:
                    # 转为灰度
                    img = img.convert("L")
                    # 转为 numpy 数组进行清洗
                    arr = np.array(img)
                    # 二值化阈值处理：消除 JPG 压缩带来的边缘模糊
                    # 大于 127 的变为 255，小于等于 127 的变为 0
                    arr = (arr > 127).astype(np.uint8) * 255
                    # 转回 PIL 图片
                    new_img = Image.fromarray(arr)
                
                # === 2. 处理 Image ===
                else:
                    # 统一转为 RGB (3通道)
                    new_img = img.convert("RGB")
                
                # === 3. 保存为 PNG ===
                # 构造新的文件名 (原文件名.png)
                new_filename = os.path.splitext(filepath)[0] + ".png"
                new_img.save(new_filename, "PNG")
                
            # === 4. 删除旧文件 (如果原文件不是 png) ===
            if ext != '.png':
                os.remove(filepath)
                
        except Exception as e:
            print(f"处理文件出错: {filepath}, 错误: {e}")

if __name__ == "__main__":
    # 1. 转换 Images (统一为 RGB PNG)
    convert_to_png(os.path.join(DATA_ROOT, "images"), is_label=False)
    
    # 2. 转换 Labels (统一为 二值化灰度 PNG)
    # convert_to_png(os.path.join(DATA_ROOT, "labels"), is_label=True)
    
    print("\n✅ 所有数据已转换为标准 PNG 格式！")