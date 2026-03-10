'''
Script usage
python src_distill/inference.py --mode distilled
python src_distill/inference.py --mode original --output_root Experiments/sam3
'''
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes

# MONAI 组件
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, 
    Resized, EnsureTyped, Lambdad
)

# Allow importing from parent directory
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src_distill.config import Config
from src_distill.model import DistillModel
from seg_utils.calculate_stats import calculate_stats_from_csv

# --- Utility Functions ---

def get_box_from_mask(mask, perturbation=0):
    """
    Generate Bounding Box (B, 4) from Mask (B, 1, H, W)
    Format: [x1, y1, x2, y2]
    """
    B, _, H, W = mask.shape
    boxes_list = []
    
    for i in range(B):
        m = mask[i, 0] # (H, W)
        if m.sum() == 0:
            # Empty mask, use full image as prompt? or center?
            # Using full image might be safer for "find anything" or just fallback
            box = torch.tensor([0, 0, W, H], device=mask.device, dtype=torch.float32)
        else:
            box = masks_to_boxes(m.unsqueeze(0))[0] # (4,)
            
            if perturbation > 0:
                noise = torch.randn_like(box) * perturbation
                box = box + noise
                box[0::2] = box[0::2].clamp(0, W)
                box[1::2] = box[1::2].clamp(0, H)
        
        boxes_list.append(box)
    
    return torch.stack(boxes_list)

class DistillInferencer:
    def __init__(self, teacher_ckpt, base_sam_ckpt, student_ckpt, device='cuda', load_student_weights=True):
        self.device = device
        print(f"🚀 Loading DistillModel on {self.device}...")
        
        # 1. Initialize Model Architecture
        # DistillModel initializes efficient-sam / sam3 internally
        self.model = DistillModel(teacher_ckpt, base_sam_ckpt).to(self.device)
        
        # 2. Load Trained Student Weights
        # self.model.student is the SAM3 component
        if load_student_weights:
            if os.path.exists(student_ckpt):
                print(f"🔥 Loading Trained Student Weights from {student_ckpt}")
                ckpt_data = torch.load(student_ckpt, map_location=self.device)
                if isinstance(ckpt_data, dict) and 'model_state_dict' in ckpt_data:
                    student_weights = ckpt_data['model_state_dict']
                else:
                    student_weights = ckpt_data
                self.model.student.load_state_dict(student_weights, strict=False) # strict=False in case of partial saving
            else:
                print(f"⚠️ Warning: Student checkpoint {student_ckpt} not found! Using base SAM3 weights.")
        else:
            print("🛡️ Using Original SAM3 Weights (No Distillation Loading)")
            
        self.model.eval()

        # 3. Pipelines (Reusing logic from UnetInferencer)
        self.transforms_with_label = Compose([
            LoadImaged(keys=["image", "label"], image_only=False, reverse_indexing=False),
            Lambdad(keys=["image"], func=self.force_rgb_cfirst),
            EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
            ScaleIntensityd(keys=["image", "label"]), 
            Resized(keys=["image", "label"], spatial_size=(512, 512), mode=["bilinear", "nearest"]),
            EnsureTyped(keys=["image", "label"])
        ])
        
        self.transforms_image_only = Compose([
            LoadImaged(keys=["image"], image_only=False, reverse_indexing=False),
            Lambdad(keys=["image"], func=self.force_rgb_cfirst),
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=(512, 512), mode="bilinear"),
            EnsureTyped(keys=["image"])
        ])

    def force_rgb_cfirst(self, img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        elif img.ndim == 3 and img.shape[-1] <= 4:
            img = np.transpose(img, (2, 0, 1))
        C, H, W = img.shape
        if C == 1:
            img = np.repeat(img, 3, axis=0)
        elif C > 3:
            img = img[:3, ...]
        return img

    def compute_dice(self, pred, gt):
        smooth = 1e-5
        intersection = (pred * gt).sum()
        return (2. * intersection + smooth) / (pred.sum() + gt.sum() + smooth)

    def compute_iou(self, pred, gt):
        smooth = 1e-5
        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum() - intersection
        return (intersection + smooth) / (union + smooth)

    def run(self, data_item, threshold=0.5, use_gt_box=False):
        """
        Inference Pipeline for Distilled SAM3
        use_gt_box: If True and label exists, use GT mask to generate box prompt (Oracle mode).
                    If False (default), use Teacher prediction to generate box prompt (Fully automated mode).
        """
        has_label = data_item.get("label") is not None
        
        # 1. Preprocess
        if has_label:
            data = self.transforms_with_label(data_item)
            gt_tensor_512 = data["label"].unsqueeze(0).to(self.device)
        else:
            temp_item = {"image": data_item["image"]}
            data = self.transforms_image_only(temp_item)
            gt_tensor_512 = None

        img_tensor_512 = data["image"].unsqueeze(0).to(self.device)

        # Get Original Size
        if "image_meta_dict" in data:
            meta = data["image_meta_dict"]
            orig_size = meta.get("spatial_shape", meta.get("dim", None))
            orig_h, orig_w = int(orig_size[1]), int(orig_size[0])
        else:
            orig_h, orig_w = img_tensor_512.shape[2], img_tensor_512.shape[3]

        # 2. Inference
        with torch.no_grad():
            # A. Generate Prompts
            if use_gt_box and has_label:
                # Oracle Prompt from GT
                boxes = get_box_from_mask(gt_tensor_512)
            else:
                # Auto Prompt from Teacher
                teacher_logits = self.model.teacher(img_tensor_512)
                teacher_mask = (torch.sigmoid(teacher_logits) > 0.5).float()
                boxes = get_box_from_mask(teacher_mask)
            
            # B. Student Forward
            # model.forward expects (images, boxes)
            # It returns (teacher_logits, student_logits)
            # Note: inside forward, images are resized to 1008x1008 for SAM3, 
            # and logits are interpolated back to input size (512x512).
            
            _, student_logits_512 = self.model(img_tensor_512, boxes=boxes)
            student_probs_512 = torch.sigmoid(student_logits_512)

        # 3. Post-process (Resize to Original)
        probs_orig = F.interpolate(
            student_probs_512, 
            size=(orig_h, orig_w), 
            mode="bilinear", 
            align_corners=False
        )
        img_tensor_orig = F.interpolate(
            img_tensor_512,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        )
        
        pred_mask_orig = (probs_orig > threshold).float()

        # Numpy conversion
        pred_np = pred_mask_orig[0, 0].cpu().numpy()
        img_np = img_tensor_orig[0].cpu().numpy().transpose(1, 2, 0)
        
        result = {
            "image": img_np,
            "pred": pred_np,
            "has_label": has_label,
            "filename": os.path.basename(data_item["image"])
        }

        # Metrics
        if has_label:
            pred_mask_512 = (student_probs_512 > threshold).float()
            pred_np_512 = pred_mask_512[0, 0].cpu().numpy()
            gt_np_512 = gt_tensor_512[0, 0].cpu().numpy()
            
            dice = self.compute_dice(pred_np_512, gt_np_512)
            iou = self.compute_iou(pred_np_512, gt_np_512)
            
            gt_tensor_orig = F.interpolate(gt_tensor_512, size=(orig_h, orig_w), mode="nearest")
            gt_np_orig = gt_tensor_orig[0, 0].cpu().numpy()
            
            result.update({
                "gt": gt_np_orig,
                "dice": dice,
                "iou": iou
            })
            
        return result

    def visualize(self, res, save_dir):
        # Same visualization logic as UnetInferencer
        filename = res['filename']
        has_label = res['has_label']
        cols = 3 if has_label else 2
        plt.figure(figsize=(4 * cols, 4))

        plt.subplot(1, cols, 1)
        plt.title(f"Image: {filename}")
        plt.imshow(res['image'], cmap="gray")
        plt.axis("off")

        if has_label:
            plt.subplot(1, cols, 2)
            plt.title("Ground Truth")
            plt.imshow(res['gt'], cmap="gray")
            plt.axis("off")
            plot_idx = 3
            title_str = f"Pred (Dice: {res['dice']:.4f})"
        else:
            plot_idx = 2
            title_str = "Prediction"

        plt.subplot(1, cols, plot_idx)
        plt.title(title_str)
        plt.imshow(res['image'], cmap="gray")
        mask_overlay = np.zeros_like(res['image'])
        mask_overlay[:, :, 0] = res['pred'] 
        plt.imshow(mask_overlay, alpha=0.4)
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"res_{filename}"))
        plt.close()

    def save_prediction_mask(self, pred_np, save_path):
        mask_uint8 = (pred_np * 255).astype(np.uint8)
        img = Image.fromarray(mask_uint8, mode='L')
        img.save(save_path)

def scan_inference_files(root_dir):
    """
    智能扫描函数：
    - 扫描 images 目录。
    - 尝试去 labels 目录找对应的 label。
    - 找不到 label 不报错，而是标记为 None。
    """
    image_dir = Path(root_dir) / "images"
    label_dir = Path(root_dir) / "labels"
    
    # 支持 png 和 jpg
    image_files = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    
    data_list = []
    
    for img_path in image_files:
        # 尝试寻找对应的 Label (假设同名，且为 png)
        # 你可以根据实际情况修改后缀匹配逻辑
        possible_label_path = label_dir / img_path.name
        
        # 检查 Label 是否存在
        label_path = str(possible_label_path) if possible_label_path.exists() else None
        
        # 解析数据集名称 (用于分组)
        dataset_name = img_path.name.split('_')[0]
        
        data_list.append({
            "image": str(img_path),
            "label": label_path, # 关键点：这里可能是 None
            "dataset_name": dataset_name
        })
        
    return data_list

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SAM3 Inference (Distilled, GT-Only, or Original)")
    parser.add_argument("--mode", type=str, default="distilled", choices=["distilled", "gtonly", "original"], 
                        help="Inference mode: 'distilled' (teacher-guided), 'gtonly' (pure GT supervision), or 'original' (base sam3)")
    parser.add_argument("--teacher_ckpt", type=str, 
                        default=Config.UNET_CKPT,
                        help="Path to Teacher U-Net checkpoint")
    parser.add_argument("--base_sam_ckpt", type=str, 
                        default=Config.SAM_CKPT,
                        help="Path to Base SAM3 checkpoint")
    parser.add_argument("--student_ckpt", type=str, 
                        default=None,
                        help="Path to Trained Student weights. If None, auto-selected based on --mode.")
    parser.add_argument("--output_root", type=str, default="inference_results_sam3", help="Root for output")
    parser.add_argument("--data_root", type=str, default=Config.DATA_ROOT, help="Root for inference data (overrides Config.DATA_ROOT)")
    
    args = parser.parse_args()

    # Determine paths and flags
    INFERENCE_ROOT = args.data_root
    
    # Logic for TARGET_DATASETS:
    # If using default Config.DATA_ROOT, we apply the default filtering (Config.EXTERNAL_TEST_DATASETS).
    # If user provides a custom --data_root, we disable filtering (process all files found).
    if args.data_root == Config.DATA_ROOT:
        TARGET_DATASETS = Config.EXTERNAL_TEST_DATASETS
    else:
        TARGET_DATASETS = None

    # Handle Student Checkpoint Selection
    student_ckpt = args.student_ckpt
    if student_ckpt is None:
        if args.mode == "distilled":
            student_ckpt = "checkpoints/student_sam3_distilled.pth"
        elif args.mode == "gtonly":
            student_ckpt = "checkpoints/student_sam3_gtonly.pth"

    # Handle Output Directory
    OUTPUT_DIR = f"{args.output_root}_{args.mode}" if args.mode != "distilled" else args.output_root
    
    LOAD_STUDENT_WEIGHTS = (args.mode in ["distilled", "gtonly"])
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"📂 Scanning files in {INFERENCE_ROOT}...")
    all_files = scan_inference_files(INFERENCE_ROOT)
    
    if TARGET_DATASETS:
        target_files = [x for x in all_files if x["dataset_name"] in TARGET_DATASETS]
    else:
        target_files = all_files
        
    print(f"🔍 Found {len(target_files)} samples.")
    
    if len(target_files) == 0:
        print("No files found! Check INFERENCE_ROOT.")
        return

    # Initialize Inferencer
    inferencer = DistillInferencer(
        teacher_ckpt=args.teacher_ckpt, 
        base_sam_ckpt=args.base_sam_ckpt, 
        student_ckpt=student_ckpt,
        load_student_weights=LOAD_STUDENT_WEIGHTS
    )
    
    results_stats = []
    
    # Inference loop
    for item in tqdm(target_files, desc="SAM3 Inference"):
        try:
            # Run with prompt from Teacher (Auto Mode)
            res = inferencer.run(item, use_gt_box=False)
            
            dataset_name = item['dataset_name']
            filename = res['filename']
            
            # Save
            overlay_dir = os.path.join(OUTPUT_DIR, dataset_name, "overlays")
            mask_dir = os.path.join(OUTPUT_DIR, dataset_name, "masks")
            os.makedirs(overlay_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            
            inferencer.visualize(res, overlay_dir)
            mask_path = os.path.join(mask_dir, filename)
            inferencer.save_prediction_mask(res['pred'], mask_path)
            
            if res['has_label']:
                results_stats.append({
                    "Dataset": dataset_name,
                    "Filename": filename,
                    "Dice": res['dice'],
                    "IoU": res['iou']
                })
        except Exception as e:
            print(f"Error processing {item['image']}: {e}")
            continue

    # Report
    if results_stats:
        df = pd.DataFrame(results_stats)
        print("\n" + "="*40)
        print("🏆 SAM3 Distilled Evaluation Report (Teacher Prompts)")
        print("="*40)
        report = df.groupby("Dataset")[["Dice", "IoU"]].agg(['mean', 'std', 'count'])
        print(report)
        metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
        df.to_csv(metrics_path, index=False)
        print(f"\n📄 Saved metrics to {metrics_path}")
        
        # Calculate and save extended statistics
        try:
            from seg_utils.calculate_stats import calculate_stats_from_csv
            stats_path = os.path.join(OUTPUT_DIR, "statistics.csv")
            print(f"Calculating extended statistics...")
            calculate_stats_from_csv(metrics_path, stats_path)
        except ImportError:
            print("Warning: Could not import seg_utils.calculate_stats. Skipping extended stats.")
        except Exception as e:
            print(f"Error calculating extended stats: {e}")
            
    else:
        print("Done. No labeled data found for metrics.")

if __name__ == "__main__":
    main()
