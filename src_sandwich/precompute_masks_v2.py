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

from src_sandwich.model_v2 import SandwichModelV2
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
    parser.add_argument("--sam_lora_ckpt", type=str,
                        default="/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/student_sam3_gtonly_lora.pth",
                        help="Path to LoRA adapter weights. Pass 'none' to use the "
                             "original (non-adapted) SAM 3.")
    parser.add_argument("--sam_probs_subdir", type=str, default="sam3_probs_lora",
                        help="Subdirectory name under data_root for SAM probability maps.")
    parser.add_argument("--data_root", type=str,
                        default="/home/maia-user/yufeng/Data/BUS_Mix_Seg")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Images per SAM forward pass. 1008x1008 SAM at bf16 "
                             "fits >=8 on a 48GB A6000.")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    sam3_probs_dir  = Path(args.data_root) / args.sam_probs_subdir
    unet1_probs_dir = Path(args.data_root) / "unet1_probs"
    sam3_probs_dir.mkdir(exist_ok=True)
    unet1_probs_dir.mkdir(exist_ok=True)

    print(f"Scanning images in {args.data_root}...")
    files = scan_images(args.data_root)
    print(f"Found {len(files)} images.")

    lora_ckpt_arg = None if args.sam_lora_ckpt.lower() == "none" else args.sam_lora_ckpt
    print(f"Loading SandwichModelV2 on {args.device} "
          f"(SAM LoRA: {lora_ckpt_arg or 'DISABLED'})...")
    model = SandwichModelV2(
        unet1_ckpt=args.teacher_ckpt,
        sam_ckpt=args.base_sam_ckpt,
        sam_lora_ckpt=lora_ckpt_arg,
    ).to(args.device)
    model.eval()

    # Filter out already-computed files up front
    todo = []
    skipped = 0
    for item in files:
        sam_save  = sam3_probs_dir  / item["filename"]
        unet_save = unet1_probs_dir / item["filename"]
        if sam_save.exists() and unet_save.exists():
            skipped += 1
        else:
            todo.append(item)
    print(f"Skipping {skipped} already-computed files; {len(todo)} remaining.")

    def _load_item(item):
        """Load, preprocess, and return (512x512 tensor, orig_h, orig_w, filename)."""
        b = transforms_image_only({"image": item["image"]})
        img = b["image"]  # (3, 512, 512)
        if "image_meta_dict" in b:
            meta = b["image_meta_dict"]
            orig_size = meta.get("spatial_shape", meta.get("dim", None))
            orig_h, orig_w = int(orig_size[1]), int(orig_size[0])
        else:
            orig_h, orig_w = img.shape[-2], img.shape[-1]
        return img, orig_h, orig_w, item["filename"]

    class _PrecomputeDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, i):
            img, h, w, fn = _load_item(self.items[i])
            return img, h, w, fn

    def _collate(batch):
        imgs = torch.stack([b[0] for b in batch], dim=0)
        hs   = [b[1] for b in batch]
        ws   = [b[2] for b in batch]
        fns  = [b[3] for b in batch]
        return imgs, hs, ws, fns

    loader = torch.utils.data.DataLoader(
        _PrecomputeDataset(todo),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=_collate,
        pin_memory=True,
    )

    pbar = tqdm(total=len(todo), desc="Precomputing soft masks")
    for imgs, hs, ws, fns in loader:
        imgs = imgs.to(args.device, non_blocking=True)  # (B, 3, 512, 512)
        try:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                unet1_logits = model.unet1(imgs)
                unet1_prob   = torch.sigmoid(unet1_logits).float()   # (B,1,512,512)
                unet1_mask   = (unet1_prob > 0.5).float()
                boxes        = model.get_box_from_mask(unet1_mask)   # (B, 4)
                sam_logits   = model.forward_sam(imgs, boxes)         # (B,1,512,512)
                sam_prob     = torch.sigmoid(sam_logits).float()
        except Exception as e:
            print(f"Error on batch starting {fns[0]}: {e}")
            pbar.update(len(fns))
            continue

        for i, fn in enumerate(fns):
            sam_save  = sam3_probs_dir  / fn
            unet_save = unet1_probs_dir / fn
            h, w = hs[i], ws[i]
            if not sam_save.exists():
                sp = F.interpolate(sam_prob[i:i+1], size=(h, w),
                                   mode="bilinear", align_corners=False)
                save_prob_as_uint8(sp, sam_save)
            if not unet_save.exists():
                up = F.interpolate(unet1_prob[i:i+1], size=(h, w),
                                   mode="bilinear", align_corners=False)
                save_prob_as_uint8(up, unet_save)
        pbar.update(len(fns))
    pbar.close()

    print(f"\nDone. Skipped {skipped} already-computed files (before run).")
    print(f"SAM3 probs:  {sam3_probs_dir}  ({len(list(sam3_probs_dir.glob('*.png')))} files)")
    print(f"UNet1 probs: {unet1_probs_dir} ({len(list(unet1_probs_dir.glob('*.png')))} files)")


if __name__ == "__main__":
    main()
