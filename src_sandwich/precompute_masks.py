import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

# Add parent directory
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src_sandwich.model import SandwichModel

# Preprocessing (Reuse identical logic from SandwichInferencer to guarantee alignment)
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
        img = img[:3, ...]
    return img

transforms_image_only = Compose([
    LoadImaged(keys=["image"], image_only=False, reverse_indexing=False),
    Lambdad(keys=["image"], func=force_rgb_cfirst),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=(512, 512), mode="bilinear"),
    EnsureTyped(keys=["image"])
])

def scan_images(root_dir):
    """
    Scan all images in DATA_ROOT.
    """
    image_dir = Path(root_dir) / "images"
    image_files = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    
    data_list = []
    for img_path in image_files:
        dataset_name = img_path.name.split('_')[0] 
        data_list.append({
            "image": str(img_path),
            "dataset_name": dataset_name,
            "filename": img_path.name
        })
    return data_list

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Precompute SAM3 Masks for Sandwich Training")
    parser.add_argument("--teacher_ckpt", type=str, 
                        default="/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/best_teacher_unet_BUSC_BUID_UDIAT_BUSUC.pth",
                        help="Path to Teacher U-Net checkpoint")
    parser.add_argument("--base_sam_ckpt", type=str, 
                        default="/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/sam3.pt",
                        help="Path to Base SAM3 checkpoint")
    parser.add_argument("--data_root", type=str, 
                        default="/home/maia-user/yufeng/Data/BUS_Mix_Seg", 
                        help="Root dataset dir containing 'images'")
    parser.add_argument("--output_dir", type=str, 
                        default="/home/maia-user/yufeng/Data/BUS_Mix_Seg/sam3_masks", 
                        help="Where to save the precomputed masks")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📂 Scanning files in {args.data_root}...")
    files = scan_images(args.data_root)
    print(f"🔍 Found {len(files)} total images to process.")

    if len(files) == 0:
        return

    # Load frozen components (We don't need U-Net2 here)
    print(f"🚀 Loading SandwichModel (Frozen Layers) on {args.device}...")
    model = SandwichModel(args.teacher_ckpt, args.base_sam_ckpt).to(args.device)
    model.eval()

    # Create dummy dir struct or dump all in one?
    # Usually safer to dump all in one folder if filenames are unique (BUSC_XXX.png)
    
    for item in tqdm(files, desc="Precomputing SAM Masks"):
        try:
            filename = item['filename']
            save_path = os.path.join(args.output_dir, filename)
            
            # Skip if already exists (Resumable)
            if os.path.exists(save_path):
                continue
                
            # Preprocess
            batch = transforms_image_only({"image": item["image"]})
            img_tensor_512 = batch["image"].unsqueeze(0).to(args.device)
            
            # Get Original Size
            if "image_meta_dict" in batch:
                meta = batch["image_meta_dict"]
                orig_size = meta.get("spatial_shape", meta.get("dim", None))
                orig_h, orig_w = int(orig_size[1]), int(orig_size[0])
            else:
                orig_h, orig_w = img_tensor_512.shape[2], img_tensor_512.shape[3]

            # Forward PASS (Only U-Net 1 and SAM3)
            with torch.no_grad():
                # U-Net 1
                unet1_logits = model.unet1(img_tensor_512)
                unet1_mask = (torch.sigmoid(unet1_logits) > 0.5).float()
                
                # Box Prompt
                boxes = model.get_box_from_mask(unet1_mask)
                
                # SAM 3
                sam_logits = model.forward_sam(img_tensor_512, boxes)
                sam_mask_512 = torch.sigmoid(sam_logits)
            
            # Resize back to original
            sam_probs_orig = F.interpolate(sam_mask_512, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            pred_sam_orig = (sam_probs_orig > 0.5).float()
            
            # Save Mask
            pred_np = pred_sam_orig[0, 0].cpu().numpy()
            mask_uint8 = (pred_np * 255).astype(np.uint8)
            Image.fromarray(mask_uint8, mode='L').save(save_path)
            
        except Exception as e:
            print(f"Error {filename}: {e}")

if __name__ == "__main__":
    main()
