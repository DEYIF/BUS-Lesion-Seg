import os
import zipfile
from tqdm import tqdm

def pack_masks_preserve_structure(source_root, output_zip_name):
    print(f"📦 正在打包: {source_root} -> {output_zip_name}")
    
    # 收集需要打包的文件列表
    files_to_pack = []
    
    # os.walk 遍历目录
    for dirpath, dirnames, filenames in os.walk(source_root):
        # 核心逻辑：只关心文件夹名字叫 'masks' 的目录
        # 这里的 dirpath 类似于: inference_results_video/benign/video_01/masks
        if os.path.basename(dirpath) == 'masks':
            for f in filenames:
                if f.endswith('.png'):
                    full_path = os.path.join(dirpath, f)
                    # 计算相对路径，以便在 zip 里保持结构
                    # 例如: benign/video_01/masks/000.png
                    rel_path = os.path.relpath(full_path, source_root)
                    files_to_pack.append((full_path, rel_path))
    
    if not files_to_pack:
        print("❌ 未找到任何 masks 文件夹或 png 文件。")
        return

    print(f"🔍 找到 {len(files_to_pack)} 个 Mask 文件，开始压缩...")

    # 开始写入 Zip
    # compression=zipfile.ZIP_DEFLATED 表示使用标准的 zip 压缩算法
    with zipfile.ZipFile(output_zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for full_path, rel_path in tqdm(files_to_pack, desc="Archiving"):
            # write(真实路径, zip里的路径)
            zf.write(full_path, arcname=rel_path)

    print(f"\n✅ 打包完成！\n文件: {output_zip_name}\n包含 masks: {len(files_to_pack)} 张")

if __name__ == "__main__":
    # 配置
    SOURCE_DIR = "/home/maia-user/yufeng/Data/WHBUS/inference"     # 你的结果根目录
    ZIP_NAME = "Data/WHBUS_masks_inference.zip" # 输出文件名
    
    pack_masks_preserve_structure(SOURCE_DIR, ZIP_NAME)