import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

# Add parent directory
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src_sandwich.model import SandwichModel
# Reuse config from src_unet or compatible
try:
    from src_unet.config import Config
except:
    pass

def scan_inference_files(root_dir):
    from pathlib import Path
    data_list = []
    # Match the logic of src_distill: assume all images in one folder or subfolders, and prefix indicates dataset.
    for root, dirs, files in os.walk(root_dir):
        if 'images' in dirs:
            img_dir = Path(root) / "images"
            lbl_dir = Path(root) / "labels"
            img_files = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
            for img_path in img_files:
                possible_lbl = lbl_dir / img_path.name
                lbl_path = str(possible_lbl) if possible_lbl.exists() else None
                # CRITICAL: Parse dataset name from prefix (e.g., BUET_xxx.png -> BUET)
                dataset_name = img_path.name.split('_')[0] 
                data_list.append({
                    "image": str(img_path),
                    "label": lbl_path,
                    "dataset_name": dataset_name
                })
    return data_list

class SandwichInferencer:
    def __init__(self, unet1_ckpt, sam_ckpt, unet2_ckpt, device='cuda',
                 sam_lora_ckpt=None):
        self.device = device
        print(f"🚀 Loading SandwichModel on {device}...")
        self.model = SandwichModel(unet1_ckpt, sam_ckpt,
                                   sam_lora_ckpt=sam_lora_ckpt).to(device)
        
        if os.path.exists(unet2_ckpt):
            print(f"🔥 Loading U-Net 2 weights from {unet2_ckpt}")
            ckpt = torch.load(unet2_ckpt, map_location=device, weights_only=False)
            sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            self.model.unet2.load_state_dict(sd)
        else:
            print(f"⚠️ U-Net 2 checkpoint not found at {unet2_ckpt}")
            
        self.model.eval()
        
        # Preprocessing (Reuse from Distill logic for robustness)
        from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, EnsureTyped, Lambdad
        
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

    def run_sample(self, data_item):
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
            unet2_logits, sam_mask_512, unet1_mask_512 = self.model(img_tensor_512)
            unet2_prob = torch.sigmoid(unet2_logits)

        # 3. Post-process (Resize to Original)
        probs_orig = F.interpolate(unet2_prob, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        sam_probs_orig = F.interpolate(sam_mask_512, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        unet1_probs_orig = F.interpolate(unet1_mask_512, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        img_tensor_orig = F.interpolate(img_tensor_512, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        
        pred_mask_orig = (probs_orig > 0.5).float()
        pred_sam_orig = (sam_probs_orig > 0.5).float()
        pred_unet1_orig = (unet1_probs_orig > 0.5).float()

        # Numpy conversion
        pred_np = pred_mask_orig[0, 0].cpu().numpy()
        pred_sam_np = pred_sam_orig[0, 0].cpu().numpy()
        pred_unet1_np = pred_unet1_orig[0, 0].cpu().numpy()
        img_np = img_tensor_orig[0].cpu().numpy().transpose(1, 2, 0)
        
        res = {
            "image": img_np,
            "pred_final": pred_np,
            "pred_sam": pred_sam_np,
            "pred_unet1": pred_unet1_np,
            "has_label": has_label,
            "filename": os.path.basename(data_item["image"])
        }

        # Metrics
        if has_label:
            pred_mask_512 = (unet2_prob > 0.5).float()
            pred_np_512 = pred_mask_512[0, 0].cpu().numpy()
            gt_np_512 = gt_tensor_512[0, 0].cpu().numpy()
            
            dice = self.compute_dice(pred_np_512, gt_np_512)
            iou = self.compute_iou(pred_np_512, gt_np_512)
            
            gt_tensor_orig = F.interpolate(gt_tensor_512, size=(orig_h, orig_w), mode="nearest")
            gt_np_orig = gt_tensor_orig[0, 0].cpu().numpy()
            
            res.update({
                "gt": gt_np_orig,
                "dice": dice,
                "iou": iou
            })
            
        return res

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sandwich Model Inference")
    parser.add_argument("--teacher_ckpt", type=str, 
                        default="/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/best_teacher_unet_BUSC_BUID_UDIAT_BUSUC.pth",
                        help="Path to Teacher U-Net checkpoint")
    parser.add_argument("--base_sam_ckpt", type=str,
                        default="/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/sam3.pt",
                        help="Path to Base SAM3 checkpoint")
    parser.add_argument("--sam_lora_ckpt", type=str, default=None,
                        help="Optional LoRA adapter checkpoint to merge into SAM3 decoder. "
                             "Pass e.g. checkpoints/student_sam3_gtonly_lora.pth for V1+LoRA inference.")
    parser.add_argument("--unet2_ckpt", type=str,
                        default="/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/sandwich_unet2.pth",
                        help="Path to Trained U-Net2 weights")
    parser.add_argument("--output_root", type=str, default="inference_results_sandwich", help="Root for output")
    parser.add_argument("--data_root", type=str, default="/home/maia-user/yufeng/Data/BUS_Mix_Seg", help="Root for inference data")
    parser.add_argument("--target_datasets", type=str,
                        default="BUSC,BUID,UDIAT,BUSUC",
                        help="Comma-separated dataset names to evaluate. Empty string = all.")
    parser.add_argument("--no_overlay", action="store_true",
                        help="Skip matplotlib overlay PNG saving (much faster). Masks are still saved.")
    
    args = parser.parse_args()

    INFERENCE_ROOT = args.data_root
    OUTPUT_DIR = args.output_root
    
    # Filter by --target_datasets (or fall back to Config if explicitly empty)
    TARGET_DATASETS = [s.strip() for s in args.target_datasets.split(",") if s.strip()]
    if not TARGET_DATASETS:
        try:
            from src_unet.config import Config
            TARGET_DATASETS = Config.EXTERNAL_TEST_DATASETS
        except:
            TARGET_DATASETS = None

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

    inferencer = SandwichInferencer(args.teacher_ckpt, args.base_sam_ckpt, args.unet2_ckpt,
                                    sam_lora_ckpt=args.sam_lora_ckpt)
    
    results_stats = []
    
    for item in tqdm(target_files, desc="Running Sandwich"):
        try:
            res = inferencer.run_sample(item)
            
            dataset_name = item['dataset_name']
            filename = res['filename']
            
            # Save visual
            mask_dir = os.path.join(OUTPUT_DIR, dataset_name, "masks")
            os.makedirs(mask_dir, exist_ok=True)

            if not args.no_overlay:
                save_dir = os.path.join(OUTPUT_DIR, dataset_name, "overlays")
                os.makedirs(save_dir, exist_ok=True)

                fig, ax = plt.subplots(1, 4, figsize=(16, 4))
                ax[0].imshow(res['image'], cmap="gray")
                ax[0].set_title("Input")

                ax[1].imshow(res['image'], cmap="gray")
                mask_overlay1 = np.zeros_like(res['image'])
                mask_overlay1[:, :, 0] = res['pred_unet1']
                ax[1].imshow(mask_overlay1, alpha=0.4)
                ax[1].set_title("Stage 1: U-Net1")

                ax[2].imshow(res['image'], cmap="gray")
                mask_overlay2 = np.zeros_like(res['image'])
                mask_overlay2[:, :, 0] = res['pred_sam']
                ax[2].imshow(mask_overlay2, alpha=0.4)
                ax[2].set_title("Stage 2: SAM3")

                ax[3].imshow(res['image'], cmap="gray")
                mask_overlay3 = np.zeros_like(res['image'])
                mask_overlay3[:, :, 0] = res['pred_final']

                if 'gt' in res:
                    mask_overlay3[:, :, 1] = res['gt']

                ax[3].imshow(mask_overlay3, alpha=0.4)

                title_final = "Final(R=Pred, G=GT)" if 'gt' in res else "Stage 3: Final"
                if 'dice' in res:
                    title_final += f"\nDice: {res['dice']:.4f}"
                ax[3].set_title(title_final, fontsize=10)

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, filename), dpi=80)
                plt.close()
            
            # Save Mask
            mask_uint8 = (res['pred_final'] * 255).astype(np.uint8)
            Image.fromarray(mask_uint8, mode='L').save(os.path.join(mask_dir, filename))
            
            # Metrics
            if res['has_label']:
                results_stats.append({
                    "Dataset": dataset_name,
                    "Filename": filename,
                    "Dice": res['dice'],
                    "IoU": res['iou']
                })
        except Exception as e:
            import traceback
            print(f"Error processing {item['image']}: {e}")
            traceback.print_exc()
            continue
            
    if results_stats:
        df = pd.DataFrame(results_stats)
        print("\n" + "="*40)
        print("🏆 Sandwich Evaluation Report")
        print("="*40)
        report = df.groupby("Dataset")[["Dice", "IoU"]].agg(['mean', 'std', 'count'])
        print(report)
        metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
        df.to_csv(metrics_path, index=False)
        print(f"\n📄 Saved metrics to {metrics_path}")
        
        try:
            from seg_utils.calculate_stats import calculate_stats_from_csv
            stats_path = os.path.join(OUTPUT_DIR, "statistics.csv")
            print(f"Calculating extended statistics...")
            calculate_stats_from_csv(metrics_path, stats_path)
        except ImportError:
            pass
    else:
        print("Done. No labeled data found for metrics.")

if __name__ == "__main__":
    main()
