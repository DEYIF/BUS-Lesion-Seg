import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityd, Resized, EnsureTyped, Lambdad
)

from src_unet.config import Config

class SimpleUnetInferencer:
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = device
        print(f"🚀 加载模型到 {self.device} ...")
        
        self.model = smp.Unet(
            encoder_name=Config.MODEL_ARCH,
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # ✅ 修复：加上 reverse_indexing=False，防止 MONAI 错误地交换 H/W 轴
        self.transforms_image_only = Compose([
            LoadImaged(keys=["image"], image_only=False, reverse_indexing=False),
            Lambdad(keys=["image"], func=self.force_rgb_cfirst),
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=Config.IMAGE_SIZE, mode="bilinear"),
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

    def run(self, image_path, threshold=0.5):
        # ✅ 稳健做法：用 PIL 先直接读原始尺寸，完全不依赖 MONAI 元数据
        with Image.open(str(image_path)) as pil_img:
            orig_w, orig_h = pil_img.size  # PIL.size 返回 (W, H)

        data_item = {"image": str(image_path)}
        data = self.transforms_image_only(data_item)
        img_tensor_512 = data["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_tensor_512)
            probs_512 = torch.sigmoid(logits)

        probs_orig = F.interpolate(
            probs_512,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        )
        pred_mask_orig = (probs_orig > threshold).float()
        pred_np = pred_mask_orig[0, 0].cpu().numpy()
        
        # 同时返回还原后的原图用于 overlay
        img_orig = F.interpolate(
            img_tensor_512,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        )
        img_np = img_orig[0].cpu().numpy().transpose(1, 2, 0)
        
        return pred_np, img_np


def save_mask(pred_np, save_path):
    mask_uint8 = (pred_np * 255).astype(np.uint8)
    img = Image.fromarray(mask_uint8, mode='L')
    img.save(save_path)


def save_overlay(img_np, pred_np, save_path):
    """在原图上叠加红色半透明 mask 并保存"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img_np, cmap="gray")
    mask_overlay = np.zeros((*pred_np.shape, 4), dtype=np.float32)
    mask_overlay[pred_np > 0] = [1.0, 0.0, 0.0, 0.4]  # 红色, 40% 透明度
    ax.imshow(mask_overlay)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def main():
    # =============== 配置 ===============
    CHECKPOINT = "/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/best_teacher_unet.pth"
    INFERENCE_ROOT = "/home/maia-user/yufeng/Data/US3M"
    OUTPUT_DIR = "/home/maia-user/yufeng/Data/US3M_masks"
    OVERLAY_DIR = "/home/maia-user/yufeng/Data/US3M_overlays"   # 新增: overlay 输出文件夹
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # ====================================

    if not os.path.exists(CHECKPOINT):
        print(f"❌ 模型文件未找到: {CHECKPOINT}")
        return

    print(f"🔍 扫描文件夹: {INFERENCE_ROOT}")
    image_paths = []
    valid_extensions = {".png", ".jpg", ".jpeg"}
    
    for root, _, files in os.walk(INFERENCE_ROOT):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions and (file.startswith("BUS_") or file.startswith("DUS_")):
                image_paths.append(os.path.join(root, file))

    print(f"共找到 {len(image_paths)} 张匹配的图像。")
    if len(image_paths) == 0:
        return

    inferencer = SimpleUnetInferencer(CHECKPOINT, device=DEVICE)
    
    print(f"💾 Mask 保存至:    {OUTPUT_DIR}")
    print(f"🖼️  Overlay 保存至: {OVERLAY_DIR}")

    for img_path_str in tqdm(image_paths, desc="推理中"):
        img_path = Path(img_path_str)
        rel_path = img_path.relative_to(INFERENCE_ROOT)

        # 保存 mask
        mask_save_path = Path(OUTPUT_DIR) / rel_path.with_suffix('.png')
        mask_save_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存 overlay（同样保持文件夹结构）
        overlay_save_path = Path(OVERLAY_DIR) / rel_path.with_suffix('.png')
        overlay_save_path.parent.mkdir(parents=True, exist_ok=True)

        pred_np, img_np = inferencer.run(img_path_str)
        save_mask(pred_np, str(mask_save_path))
        save_overlay(img_np, pred_np, str(overlay_save_path))

    print("✅ 推理全部完成！")


if __name__ == "__main__":
    main()
