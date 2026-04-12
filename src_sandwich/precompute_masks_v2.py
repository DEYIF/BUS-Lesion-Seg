"""
Precompute soft probability masks for Sandwich V2 training.

Unlike precompute_masks.py (which thresholds SAM output to binary),
this script saves:
  - sam3_probs/   — SAM3 sigmoid probability (0–255 uint8, representing 0.0–1.0)
  - unet1_probs/  — UNet1 sigmoid probability (0–255 uint8, representing 0.0–1.0)

Saving probabilities instead of binary masks fixes the train/inference
distribution mismatch: during inference model.forward() passes soft
sigmoid probabilities to the Student; now training matches.

Usage:
    conda run -n sam3 python src_sandwich/precompute_masks_v2.py
"""

import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src_sandwich.model import SandwichModel
from monai.transforms import Compose, LoadImaged, ScaleIntensityd, Resized, EnsureTyped, Lambdad


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


transforms_image_only = Compose([
    LoadImaged(keys=["image"], image_only=False, reverse_indexing=False),
    Lambdad(keys=["image"], func=force_rgb_cfirst),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=(512, 512), mode="bilinear"),
    EnsureTyped(keys=["image"])
])


def scan_images(root_dir):
    image_dir = Path(root_dir) / "images"
    image_files = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    return [{"image": str(p), "filename": p.name} for p in image_files]


def save_prob_as_uint8(prob_tensor, save_path):
    """Save a (1, H, W) probability tensor [0,1] as uint8 grayscale PNG [0,255]."""
    arr = prob_tensor[0, 0].cpu().numpy()          # (H, W) float32 in [0,1]
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr, mode='L').save(save_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", type=str,
                        default="/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/best_teacher_unet_BUSC_BUID_UDIAT_BUSUC.pth")
    parser.add_argument("--base_sam_ckpt", type=str,
                        default="/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/sam3.pt")
    parser.add_argument("--data_root", type=str,
                        default="/home/maia-user/yufeng/Data/BUS_Mix_Seg")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    sam3_probs_dir  = Path(args.data_root) / "sam3_probs"
    unet1_probs_dir = Path(args.data_root) / "unet1_probs"
    sam3_probs_dir.mkdir(exist_ok=True)
    unet1_probs_dir.mkdir(exist_ok=True)

    print(f"Scanning images in {args.data_root}...")
    files = scan_images(args.data_root)
    print(f"Found {len(files)} images.")

    print(f"Loading SandwichModel on {args.device}...")
    model = SandwichModel(args.teacher_ckpt, args.base_sam_ckpt).to(args.device)
    model.eval()

    skipped = 0
    for item in tqdm(files, desc="Precomputing soft masks"):
        filename = item["filename"]
        sam_save  = sam3_probs_dir  / filename
        unet_save = unet1_probs_dir / filename

        # Both already exist → skip
        if sam_save.exists() and unet_save.exists():
            skipped += 1
            continue

        try:
            batch = transforms_image_only({"image": item["image"]})
            img_tensor = batch["image"].unsqueeze(0).to(args.device)

            # Original size for upsampling back
            if "image_meta_dict" in batch:
                meta = batch["image_meta_dict"]
                orig_size = meta.get("spatial_shape", meta.get("dim", None))
                orig_h, orig_w = int(orig_size[1]), int(orig_size[0])
            else:
                orig_h, orig_w = img_tensor.shape[2], img_tensor.shape[3]

            with torch.no_grad():
                # UNet1 soft probability
                unet1_logits = model.unet1(img_tensor)
                unet1_prob   = torch.sigmoid(unet1_logits)          # (1,1,512,512)

                # SAM3 soft probability (NO threshold)
                unet1_mask = (unet1_prob > 0.5).float()
                boxes      = model.get_box_from_mask(unet1_mask)
                sam_logits = model.forward_sam(img_tensor, boxes)
                sam_prob   = torch.sigmoid(sam_logits)               # (1,1,512,512)

            # Resize back to original resolution
            unet1_prob_orig = F.interpolate(unet1_prob, size=(orig_h, orig_w),
                                            mode="bilinear", align_corners=False)
            sam_prob_orig   = F.interpolate(sam_prob,   size=(orig_h, orig_w),
                                            mode="bilinear", align_corners=False)

            if not unet_save.exists():
                save_prob_as_uint8(unet1_prob_orig, unet_save)
            if not sam_save.exists():
                save_prob_as_uint8(sam_prob_orig, sam_save)

        except Exception as e:
            print(f"Error on {filename}: {e}")

    print(f"\nDone. Skipped {skipped} already-computed files.")
    print(f"SAM3 probs:  {sam3_probs_dir}  ({len(list(sam3_probs_dir.glob('*.png')))} files)")
    print(f"UNet1 probs: {unet1_probs_dir} ({len(list(unet1_probs_dir.glob('*.png')))} files)")


if __name__ == "__main__":
    main()
