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
    def __init__(self, unet1_ckpt, sam_ckpt, unet2_ckpt, device='cuda',
                 sam_lora_ckpt=None):
        self.device = device
        print(f"Loading SandwichModelV2 on {device}...")
        self.model = SandwichModelV2(
            unet1_ckpt, sam_ckpt,
            sam_lora_ckpt=sam_lora_ckpt,
        ).to(device)

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

    def run_sample(self, data_item, a1_cfg=None, fallback_mode="legacy"):
        """
        a1_cfg: optional dict with keys
            v1_mask_dir, a1_v2_v1_ratio, a1_v1_min_area
        If provided AND the trigger fires, V2's final mask is replaced with V1's.
        Only used when fallback_mode == "legacy".

        fallback_mode:
            "legacy" — empty student → UNet1 soft; A1 oversize → V1+LoRA mask (external).
            "swap"   — empty student → SAM^LoRA binary; oversize vs SAM^LoRA → UNet1 soft.
                       Fully self-contained: does not read external v1_mask_dir.
        """
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

        if fallback_mode == "legacy":
            # ── Adaptive fallback (existing): empty student → UNet1 ────────────
            student_empty = ((unet2_prob > 0.5).float().sum() == 0)
            if student_empty:
                final_prob_512 = unet1_soft    # fall back to UNet1
                used_fallback  = True
            else:
                final_prob_512 = unet2_prob
                used_fallback  = False

            # ── A1 fallback: V2 area >> V1+LoRA area → use V1 mask ─────────────
            used_a1 = False
            if a1_cfg is not None and not used_fallback:
                ds_name  = data_item.get("dataset_name") or os.path.basename(data_item["image"]).split("_")[0]
                filename = os.path.basename(data_item["image"])
                v1_path  = os.path.join(a1_cfg["v1_mask_dir"], ds_name, "masks", filename)
                if os.path.exists(v1_path):
                    v1_arr = np.array(Image.open(v1_path).convert("L"))
                    if v1_arr.shape != (512, 512):
                        v1_arr = np.array(
                            Image.fromarray(v1_arr).resize((512, 512), Image.NEAREST)
                        )
                    v1_bin = (v1_arr > 127).astype(np.float32)

                    v2_bin   = (final_prob_512 > 0.5).float()[0, 0].cpu().numpy()
                    v2_area  = float(v2_bin.sum())
                    v1_area  = float(v1_bin.sum())
                    img_size = float(v2_bin.size)

                    if v1_area > a1_cfg["a1_v1_min_area"] * img_size and \
                       v2_area > a1_cfg["a1_over_ratio"] * v1_area:
                        final_prob_512 = torch.from_numpy(v1_bin).float()[None, None].to(self.device)
                        used_a1 = True

        elif fallback_mode == "swap":
            # ── Empty fallback: empty student → SAM^LoRA binary ─────────────
            used_fallback = False
            used_a1       = False  # reused: records oversize→UNet1 trigger in swap mode

            student_empty = ((unet2_prob > 0.5).float().sum() == 0)
            if student_empty:
                final_prob_512 = (sam_soft > 0.5).float()
                used_fallback  = True
            else:
                final_prob_512 = unet2_prob

            # ── Oversize fallback: V2 area > 1.5*SAM area & SAM>=1% → UNet1 soft ──
            if not used_fallback and a1_cfg is not None:
                sam_bin_np = (sam_soft > 0.5).float()[0, 0].cpu().numpy()
                v2_bin_np  = (final_prob_512 > 0.5).float()[0, 0].cpu().numpy()
                sam_area   = float(sam_bin_np.sum())
                v2_area    = float(v2_bin_np.sum())
                img_size   = float(sam_bin_np.size)

                if sam_area > a1_cfg["a1_v1_min_area"] * img_size and \
                   v2_area > a1_cfg["a1_over_ratio"] * sam_area:
                    final_prob_512 = unet1_soft
                    used_a1 = True
        else:
            raise ValueError(f"Unknown fallback_mode: {fallback_mode}")
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
            "used_a1":      used_a1,
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
    parser.add_argument("--sam_lora_ckpt", type=str, default="none",
                        help="LoRA adapter weights. 'none' to disable.")
    parser.add_argument("--output_root", type=str, default="Experiments/sandwich_v2")
    parser.add_argument("--data_root", type=str,
                        default="/home/maia-user/yufeng/Data/BUS_Mix_Seg")
    parser.add_argument("--target_datasets", type=str, default="BUSC,BUID,UDIAT,BUSUC",
                        help="Comma-separated dataset names for evaluation.")
    # ── A1 (V2-vs-V1 over-segmentation fallback, rule R6) ─────────────────
    # Rule (default R6): use V1 if v2_area > over_ratio * v1_area AND
    #                                v1_area > min_area * image_size.
    # Rationale: empirically the dominant V2 retreat mode is OVER-segmentation
    # (V2 mask is much LARGER than V1 mask while V1 is reasonably confident).
    # See ReportsCC/2026-04-29 report for offline rule sweep.
    parser.add_argument("--enable_a1", action="store_true",
                        help="Enable A1 fallback: when V2 mask area is much LARGER "
                             "than V1+LoRA mask area (and V1 is confident), use V1 mask.")
    parser.add_argument("--v1_mask_dir", type=str,
                        default="Experiments/sandwich_v1_lora",
                        help="Directory holding V1+LoRA per-dataset masks "
                             "(<dataset>/masks/<filename>.png)")
    parser.add_argument("--a1_over_ratio", type=float, default=1.5,
                        help="Trigger A1 if v2_area > over_ratio * v1_area (default 1.5).")
    parser.add_argument("--a1_v1_min_area", type=float, default=0.01,
                        help="Trigger A1 only if v1_area > min_area * image_size "
                             "(default 0.01 = 1% of image, filters out tiny V1 masks).")
    parser.add_argument("--no_overlay", action="store_true",
                        help="Skip matplotlib overlay PNG saving for speed.")
    parser.add_argument("--fallback_mode", type=str, default="legacy",
                        choices=["legacy", "swap"],
                        help="legacy = current logic (empty→UNet1 soft, A1→V1+LoRA mask); "
                             "swap = experimental (empty→SAM^LoRA binary, oversize vs SAM→UNet1 soft).")
    args = parser.parse_args()

    OUTPUT_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        args.output_root
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lora_ckpt = None if args.sam_lora_ckpt.lower() == "none" else args.sam_lora_ckpt

    TARGET_DATASETS = [s.strip() for s in args.target_datasets.split(",") if s.strip()]

    all_files = scan_inference_files(args.data_root)
    target_files = [x for x in all_files if x["dataset_name"] in TARGET_DATASETS] \
        if TARGET_DATASETS else all_files

    print(f"Found {len(target_files)} samples from {TARGET_DATASETS}.")
    inferencer = SandwichV2Inferencer(args.teacher_ckpt, args.base_sam_ckpt,
                                      args.unet2_ckpt, device=device,
                                      sam_lora_ckpt=lora_ckpt)

    a1_cfg = None
    if args.fallback_mode == "legacy" and args.enable_a1:
        v1_dir = args.v1_mask_dir
        if not os.path.isabs(v1_dir):
            v1_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), v1_dir)
        a1_cfg = {
            "v1_mask_dir":    v1_dir,
            "a1_over_ratio":  args.a1_over_ratio,
            "a1_v1_min_area": args.a1_v1_min_area,
        }
        print(f"[legacy] A1 fallback enabled. v1_mask_dir={v1_dir}, "
              f"trigger if v2_area > {args.a1_over_ratio} * v1_area "
              f"AND v1_area > {args.a1_v1_min_area} * image_size")
    elif args.fallback_mode == "swap":
        # swap 模式下 oversize fallback 默认启用，参照源是内部 sam_soft，不读 v1_mask_dir
        a1_cfg = {
            "a1_over_ratio":  args.a1_over_ratio,
            "a1_v1_min_area": args.a1_v1_min_area,
        }
        print(f"[swap] empty→SAM^LoRA binary; oversize→UNet1 soft "
              f"(trigger if v2_area > {args.a1_over_ratio} * sam_area "
              f"AND sam_area > {args.a1_v1_min_area} * image_size)")

    results_stats = []
    fallback_count = 0
    a1_count = 0

    for item in tqdm(target_files, desc="Sandwich V2"):
        try:
            res = inferencer.run_sample(item, a1_cfg=a1_cfg,
                                         fallback_mode=args.fallback_mode)
            dataset_name = item["dataset_name"]
            filename = res["filename"]
            if res["used_fallback"]:
                fallback_count += 1
            if res.get("used_a1"):
                a1_count += 1

            mask_dir = os.path.join(OUTPUT_DIR, dataset_name, "masks")
            os.makedirs(mask_dir, exist_ok=True)

            if not args.no_overlay:
                save_dir = os.path.join(OUTPUT_DIR, dataset_name, "overlays")
                os.makedirs(save_dir, exist_ok=True)

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
                if res.get("used_a1"):
                    title += " [A1]"
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

    if args.fallback_mode == "legacy":
        print(f"\n[legacy] Empty-student fallback (→UNet1 soft): {fallback_count} / {len(target_files)}")
        if args.enable_a1:
            print(f"[legacy] A1 fallback (→V1+LoRA mask): {a1_count} / {len(target_files)}")
    else:  # swap
        print(f"\n[swap] Empty-student fallback (→SAM^LoRA binary): {fallback_count} / {len(target_files)}")
        print(f"[swap] Oversize fallback (→UNet1 soft):        {a1_count} / {len(target_files)}")

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
