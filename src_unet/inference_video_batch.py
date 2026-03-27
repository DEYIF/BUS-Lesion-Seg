import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

try:
    from src_unet.inference import UnetInferencer
    from src_unet.config import Config
except ImportError:
    from inference import UnetInferencer
    from config import Config

class VideoFrameDataset(Dataset):
    def __init__(self, tasks, transform):
        self.tasks = tasks
        self.transform = transform
        
    def __len__(self):
        return len(self.tasks)
        
    def __getitem__(self, idx):
        item = self.tasks[idx]
        # ✅ 稳健做法：用 PIL 先读原始尺寸，不依赖 MONAI metadata
        with Image.open(item["image"]) as pil_img:
            orig_w, orig_h = pil_img.size  # PIL.size 返回 (W, H)
            
        # MONAI transforms expect dictionary
        temp_item = {"image": item["image"]}
        data = self.transform(temp_item)
            
        return {
            "image": data["image"],
            "orig_h": orig_h,
            "orig_w": orig_w,
            "meta_category": item["meta_category"],
            "meta_video": item["meta_video"],
            "meta_filename": item["meta_filename"]
        }

def scan_video_dataset(root_dir):
    root = Path(root_dir)
    all_frames = sorted(list(root.rglob("*.png")))
    tasks = []
    for frame_path in all_frames:
        category = frame_path.parts[-3]
        video_name = frame_path.parts[-2]
        filename = frame_path.name
        tasks.append({
            "image": str(frame_path),
            "meta_category": category,
            "meta_video": video_name,
            "meta_filename": filename
        })
    return tasks

def save_prediction_mask(pred_np, save_path):
    mask_uint8 = (pred_np * 255).astype(np.uint8)
    img = Image.fromarray(mask_uint8, mode='L')
    img.save(save_path)

def main():
    VIDEO_DATA_ROOT = "/home/maia-user/yufeng/Data/WHBUS/buvimgs-001" 
    OUTPUT_ROOT = "/home/maia-user/yufeng/Data/WHBUS/masks_inference"
    # ✅ 修复：指向正确的模型文件
    CHECKPOINT = os.path.join(Config.CHECKPOINT_DIR, "best_teacher_unet.pth")
    BATCH_SIZE = Config.BATCH_SIZE * 2
    # 只为这个指定的子文件夹输出 overlay，方便检查；设为 None 则全部输出
    OVERLAY_VIDEO = "benign_9161001_10"  # 实际存在的第一个视频文件夹名
    
    if not os.path.exists(VIDEO_DATA_ROOT):
        print(f"❌ 找不到数据目录: {VIDEO_DATA_ROOT}")
        return

    print(f"🚀 初始化推理器: {CHECKPOINT}")
    inferencer = UnetInferencer(CHECKPOINT)
    
    print(f"📂 正在扫描视频帧目录: {VIDEO_DATA_ROOT} ...")
    tasks = scan_video_dataset(VIDEO_DATA_ROOT)
    print(f"🔍 共找到 {len(tasks)} 帧待处理")
    
    # Create dataset and dataloader
    dataset = VideoFrameDataset(tasks, transform=inferencer.transforms_image_only)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"⚡ 开始批量推理 (Batch Size: {BATCH_SIZE})...")
    
    for batch in tqdm(dataloader, desc="Batch Inference"):
        images = batch["image"].to(inferencer.device)
        orig_hs = batch["orig_h"].numpy()
        orig_ws = batch["orig_w"].numpy()
        categories = batch["meta_category"]
        videos = batch["meta_video"]
        filenames = batch["meta_filename"]
        
        with torch.no_grad():
            logits = inferencer.model(images)
            probs_512 = torch.sigmoid(logits) # (B, 1, 512, 512)
            
        # Iterate over items in the batch
        for i in range(len(images)):
            h, w = int(orig_hs[i]), int(orig_ws[i])
            if h == 0 or w == 0:
                h, w = images[i].shape[1], images[i].shape[2]
                
            prob_single = probs_512[i:i+1] # (1, 1, 512, 512)
            
            # Interpolate to original size
            probs_orig = F.interpolate(
                prob_single, 
                size=(h, w), 
                mode="bilinear", 
                align_corners=False
            )
            
            pred_mask_orig = (probs_orig > 0.5).float()
            pred_np = pred_mask_orig[0, 0].cpu().numpy()
            
            # Save
            cat = categories[i]
            vid = videos[i]
            fname = filenames[i]
            
            mask_save_dir = os.path.join(OUTPUT_ROOT, cat, vid)
            os.makedirs(mask_save_dir, exist_ok=True)
            mask_path = os.path.join(mask_save_dir, fname)
            
            save_prediction_mask(pred_np, mask_path)

            # 只为 OVERLAY_VIDEO 指定的子文件夹输出 overlay（None 则全部输出）
            if OVERLAY_VIDEO is None or vid == OVERLAY_VIDEO:
                overlay_save_dir = os.path.join(OUTPUT_ROOT, cat, vid, "overlays")
                os.makedirs(overlay_save_dir, exist_ok=True)
                
                img_single = images[i:i+1]
                img_orig = F.interpolate(
                    img_single,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False
                )
                img_np = img_orig[0].cpu().numpy().transpose(1, 2, 0)
                
                res_dict = {
                    "filename": fname,
                    "pred": pred_np,
                    "image": img_np,
                    "has_label": False
                }
                inferencer.visualize(res_dict, overlay_save_dir)

    print(f"\n✅ 视频批量推理完成！结果已保存在: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
