"""
Sandwich V2 Inference — 5-channel Student UNet with adaptive fallback.

Uses SandwichModelV2 (model_v2.py):
  Input to Student: [RGB, soft_SAM, soft_UNet1]  (5 channels)

Post-processing:
  If Student predicts empty mask → fall back to UNet1 prediction.

Usage:
    conda run -n sam3 python src_sandwich/inference_v2.py \
        --unet2_ckpt checkpoints/sandwich_unet2_v2.pth \
        --output_root Experiments/sandwich_v2
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src_sandwich.model_v2 import SandwichModelV2


def scan_inference_files(root_dir):
    from pathlib import Path
    data_list = []
    for root, dirs, files in os.walk(root_dir):
        if 'images' in dirs:
            img_dir = Path(root) / "images"
            lbl_dir = Path(root) / "labels"
            img_files = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
            for img_path in img_files:
                possible_lbl = lbl_dir / img_path.name
                lbl_path = str(possible_lbl) if possible_lbl.exists() else None
                dataset_name = img_path.name.split('_')[0]
                data_list.append({
                    "image": str(img_path),
                    "label": lbl_path,
                    "dataset_name": dataset_name
                })
    return data_list


class SandwichV2Inferencer:
    def __init__(self, unet1_ckpt, sam_ckpt, unet2_ckpt, device='cuda'):
        self.device = device
        print(f"Loading SandwichModelV2 on {device}...")
        self.model = SandwichModelV2(unet1_ckpt, sam_ckpt).to(device)

        if os.path.exists(unet2_ckpt):
            print(f"Loading V2 U-Net 2 weights from {unet2_ckpt}")
            ckpt = torch.load(unet2_ckpt, map_location=device, weights_only=False)
            state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model.unet2.load_state_dict(state)
        else:
            print(f"Warning: checkpoint not found at {unet2_ckpt}")

        self.model.eval()

        from monai.transforms import (
            Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
            Resized, EnsureTyped, Lambdad
        )

        def force_rgb_cfirst(img):
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
                img = img[:3]
            return img

        self.transforms_with_label = Compose([
            LoadImaged(keys=["image", "label"], image_only=False, reverse_indexing=False),
            Lambdad(keys=["image"], func=force_rgb_cfirst),
            EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
            ScaleIntensityd(keys=["image", "label"]),
            Resized(keys=["image", "label"], spatial_size=(512, 512),
                    mode=["bilinear", "nearest"]),
            EnsureTyped(keys=["image", "label"])
        ])
        self.transforms_image_only = Compose([
            LoadImaged(keys=["image"], image_only=False, reverse_indexing=False),
            Lambdad(keys=["image"], func=force_rgb_cfirst),
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=(512, 512), mode="bilinear"),
            EnsureTyped(keys=["image"])
        ])

    def compute_dice(self, pred, gt):
        smooth = 1e-5
        return (2. * (pred * gt).sum() + smooth) / (pred.sum() + gt.sum() + smooth)

    def compute_iou(self, pred, gt):
        smooth = 1e-5
        i = (pred * gt).sum()
        u = pred.sum() + gt.sum() - i
        return (i + smooth) / (u + smooth)

    def run_sample(self, data_item):
        has_label = data_item.get("label") is not None

        if has_label:
            data = self.transforms_with_label(data_item)
            gt_tensor_512 = data["label"].unsqueeze(0).to(self.device)
        else:
            data = self.transforms_image_only({"image": data_item["image"]})
            gt_tensor_512 = None

        img_tensor_512 = data["image"].unsqueeze(0).to(self.device)

        if "image_meta_dict" in data:
            meta = data["image_meta_dict"]
            orig_size = meta.get("spatial_shape", meta.get("dim", None))
            orig_h, orig_w = int(orig_size[1]), int(orig_size[0])
        else:
            orig_h, orig_w = img_tensor_512.shape[2], img_tensor_512.shape[3]

        with torch.no_grad():
            unet2_logits, sam_soft, unet1_soft, unet1_mask = self.model(img_tensor_512)
            unet2_prob = torch.sigmoid(unet2_logits)

        # ── Adaptive fallback ──────────────────────────────────────────────
        student_empty = ((unet2_prob > 0.5).float().sum() == 0)
        if student_empty:
            final_prob_512 = unet1_soft    # fall back to UNet1
            used_fallback  = True
        else:
            final_prob_512 = unet2_prob
            used_fallback  = False
        # ──────────────────────────────────────────────────────────────────

        probs_orig    = F.interpolate(final_prob_512, size=(orig_h, orig_w),
                                      mode="bilinear", align_corners=False)
        sam_orig      = F.interpolate(sam_soft,       size=(orig_h, orig_w),
                                      mode="bilinear", align_corners=False)
        unet1_orig    = F.interpolate(unet1_mask,     size=(orig_h, orig_w),
                                      mode="bilinear", align_corners=False)
        img_orig      = F.interpolate(img_tensor_512, size=(orig_h, orig_w),
                                      mode="bilinear", align_corners=False)

        pred_np    = (probs_orig > 0.5).float()[0, 0].cpu().numpy()
        pred_sam_np = (sam_orig > 0.5).float()[0, 0].cpu().numpy()
        pred_u1_np  = (unet1_orig > 0.5).float()[0, 0].cpu().numpy()
        img_np      = img_orig[0].cpu().numpy().transpose(1, 2, 0)

        res = {
            "image":        img_np,
            "pred_final":   pred_np,
            "pred_sam":     pred_sam_np,
            "pred_unet1":   pred_u1_np,
            "used_fallback": used_fallback,
            "has_label":    has_label,
            "filename":     os.path.basename(data_item["image"])
        }

        if has_label:
            pred_512 = (final_prob_512 > 0.5).float()
            res["dice"] = self.compute_dice(pred_512[0,0].cpu().numpy(),
                                            gt_tensor_512[0,0].cpu().numpy())
            res["iou"]  = self.compute_iou(pred_512[0,0].cpu().numpy(),
                                            gt_tensor_512[0,0].cpu().numpy())
            gt_orig = F.interpolate(gt_tensor_512, size=(orig_h, orig_w), mode="nearest")
            res["gt"] = gt_orig[0, 0].cpu().numpy()

        return res


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sandwich V2 Inference")
    parser.add_argument("--teacher_ckpt", type=str,
                        default="/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/best_teacher_unet_BUSC_BUID_UDIAT_BUSUC.pth")
    parser.add_argument("--base_sam_ckpt", type=str,
                        default="/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/sam3.pt")
    parser.add_argument("--unet2_ckpt", type=str,
                        default="/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/sandwich_unet2_v2.pth")
    parser.add_argument("--output_root", type=str, default="Experiments/sandwich_v2")
    parser.add_argument("--data_root", type=str,
                        default="/home/maia-user/yufeng/Data/BUS_Mix_Seg")
    args = parser.parse_args()

    OUTPUT_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        args.output_root
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        from src_unet.config import Config
        TARGET_DATASETS = Config.EXTERNAL_TEST_DATASETS
    except Exception:
        TARGET_DATASETS = None

    all_files = scan_inference_files(args.data_root)
    target_files = [x for x in all_files if x["dataset_name"] in TARGET_DATASETS] \
        if TARGET_DATASETS else all_files

    print(f"Found {len(target_files)} samples.")
    inferencer = SandwichV2Inferencer(args.teacher_ckpt, args.base_sam_ckpt,
                                      args.unet2_ckpt, device=device)

    results_stats = []
    fallback_count = 0

    for item in tqdm(target_files, desc="Sandwich V2"):
        try:
            res = inferencer.run_sample(item)
            dataset_name = item["dataset_name"]
            filename = res["filename"]
            if res["used_fallback"]:
                fallback_count += 1

            save_dir = os.path.join(OUTPUT_DIR, dataset_name, "overlays")
            mask_dir = os.path.join(OUTPUT_DIR, dataset_name, "masks")
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(res["image"], cmap="gray"); axes[0].set_title("Input"); axes[0].axis("off")

            axes[1].imshow(res["image"], cmap="gray")
            ov1 = np.zeros_like(res["image"]); ov1[:,:,0] = res["pred_unet1"]
            axes[1].imshow(ov1, alpha=0.4); axes[1].set_title("Stage1 UNet"); axes[1].axis("off")

            axes[2].imshow(res["image"], cmap="gray")
            ov2 = np.zeros_like(res["image"]); ov2[:,:,0] = res["pred_sam"]
            axes[2].imshow(ov2, alpha=0.4); axes[2].set_title("Stage2 SAM"); axes[2].axis("off")

            axes[3].imshow(res["image"], cmap="gray")
            ov3 = np.zeros_like(res["image"]); ov3[:,:,0] = res["pred_final"]
            if "gt" in res:
                ov3[:,:,1] = res["gt"]
            axes[3].imshow(ov3, alpha=0.4)
            title = "V2 Final"
            if res["used_fallback"]:
                title += " [fallback]"
            if "dice" in res:
                title += f"\nDice={res['dice']:.4f}"
            axes[3].set_title(title, fontsize=9); axes[3].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, filename), dpi=80)
            plt.close()

            mask_uint8 = (res["pred_final"] * 255).astype(np.uint8)
            Image.fromarray(mask_uint8, mode="L").save(os.path.join(mask_dir, filename))

            if res["has_label"]:
                results_stats.append({
                    "Dataset":  dataset_name,
                    "Filename": filename,
                    "Dice":     res["dice"],
                    "IoU":      res["iou"],
                })

        except Exception as e:
            import traceback
            print(f"Error: {item['image']}: {e}")
            traceback.print_exc()

    print(f"\nFallback used: {fallback_count} / {len(target_files)}")

    if results_stats:
        df = pd.DataFrame(results_stats)
        metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
        df.to_csv(metrics_path, index=False)

        print("\n" + "=" * 50)
        print("Sandwich V2 Evaluation")
        print("=" * 50)
        report = df.groupby("Dataset")[["Dice", "IoU"]].mean()
        print(report)
        print(f"\nOverall: Dice={df['Dice'].mean():.4f}  IoU={df['IoU'].mean():.4f}")

        try:
            from seg_utils.calculate_stats import calculate_stats_from_csv
            calculate_stats_from_csv(metrics_path, os.path.join(OUTPUT_DIR, "statistics.csv"))
        except ImportError:
            pass


if __name__ == "__main__":
    main()
