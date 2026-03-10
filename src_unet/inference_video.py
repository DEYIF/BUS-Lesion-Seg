import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

# === 关键：复用你之前写好的类 ===
# 这样你就不用把模型定义、预处理再写一遍了
try:
    from src_unet.inference import UnetInferencer
    from src_unet.config import Config
except ImportError:
    from inference import UnetInferencer
    from config import Config

def scan_video_dataset(root_dir):
    """
    专门针对视频数据集的扫描器
    结构: root / category (benign/malignant) / video_name / frames.png
    """
    root = Path(root_dir)
    
    # 查找所有 png 文件 (递归查找)
    # rglog('*/*.png') 会匹配所有子文件夹里的 png
    all_frames = sorted(list(root.rglob("*.png")))
    
    tasks = []
    for frame_path in all_frames:
        # 获取路径各部分
        # 例如: Data/buvimgs-001/benign/benign_9161001_10/000000.png
        
        # parts[-3] -> benign
        # parts[-2] -> benign_9161001_10
        # parts[-1] -> 000000.png
        category = frame_path.parts[-3]
        video_name = frame_path.parts[-2]
        filename = frame_path.name
        
        # 构造任务字典
        tasks.append({
            "image": str(frame_path),
            "label": None, # 视频推理通常没有 Label
            "dataset_name": "BUV_Video", # 统称
            # 额外记录元数据，方便保存时重建目录
            "meta_category": category,
            "meta_video": video_name,
            "meta_filename": filename
        })
        
    return tasks

def main():
    # === ⚙️ 配置区 ===
    # 视频数据的根目录
    VIDEO_DATA_ROOT = "/home/maia-user/yufeng/Data/WHBUS/buvimgs-001" 
    
    # 结果保存目录
    OUTPUT_ROOT = "/home/maia-user/yufeng/Data/WHBUS/masks_inference"
    
    # 模型路径
    CHECKPOINT = os.path.join(Config.CHECKPOINT_DIR, "best_teacher_unet_allSeg.pth")
    # =================
    
    if not os.path.exists(VIDEO_DATA_ROOT):
        print(f"❌ 找不到数据目录: {VIDEO_DATA_ROOT}")
        return

    # 1. 初始化推理器 (直接复用)
    print(f"🚀 初始化推理器: {CHECKPOINT}")
    inferencer = UnetInferencer(CHECKPOINT)
    
    # 2. 扫描文件
    print(f"📂 正在扫描视频帧目录: {VIDEO_DATA_ROOT} ...")
    tasks = scan_video_dataset(VIDEO_DATA_ROOT)
    print(f"🔍 共找到 {len(tasks)} 帧待处理")
    
    # 3. 循环推理
    for item in tqdm(tasks, desc="Processing Video Frames"):
        try:
            # 运行推理 (复用 run 方法)
            # run 方法会自动处理无 label 的情况
            res = inferencer.run(item)
            
            # === 📁 重建目录结构 ===
            # 我们希望结果存为: output / benign / benign_916... / 000000.png
            category = item['meta_category']
            video_name = item['meta_video']
            filename = item['meta_filename']
            
            # 构建保存路径
            # 1. 保存 Mask (纯黑白)
            # mask_save_dir = os.path.join(OUTPUT_ROOT, category, video_name, "masks")
            mask_save_dir = os.path.join(OUTPUT_ROOT, category, video_name)
            os.makedirs(mask_save_dir, exist_ok=True)
            
            mask_path = os.path.join(mask_save_dir, filename)
            inferencer.save_prediction_mask(res['pred'], mask_path)
            
            # # 2. (可选) 保存 Overlay 可视化图
            # # 如果不想每帧都存叠加图(太占空间)，可以注释掉下面几行
            # overlay_save_dir = os.path.join(OUTPUT_ROOT, category, video_name, "overlays")
            # os.makedirs(overlay_save_dir, exist_ok=True)
            
            # # 为了配合 visualize 接口，我们构造一个结果字典
            # # 因为 visualize 里面是用 res['filename'] 来命名的，我们不需要改动它
            # inferencer.visualize(res, overlay_save_dir)

        except Exception as e:
            print(f"⚠️ 处理出错: {item['image']} | Error: {e}")
            continue

    print(f"\n✅ 视频推理完成！结果已保存在: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()