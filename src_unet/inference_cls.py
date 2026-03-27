import os
import torch
import numpy as np
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
    def __init__(self, checkpoint_path, device=None):
        self.device = device if device else Config.DEVICE
        print(f"🚀 加载模型到 {self.device} ...")
        
        # 1. 加载模型结构
        self.model = smp.Unet(
            encoder_name=Config.MODEL_ARCH,
            encoder_weights=None, 
            in_channels=3,
            classes=1,
        )
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # ✅ 修复：加上 reverse_indexing=False
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
        # ✅ 稳健做法：用 PIL 在任何变换之前直接读原始尺寸
        with Image.open(str(image_path)) as pil_img:
            orig_w, orig_h = pil_img.size  # PIL.size 返回 (W, H)

        # 1. 预处理
        data_item = {"image": str(image_path)}
        data = self.transforms_image_only(data_item)
        img_tensor_512 = data["image"].unsqueeze(0).to(self.device)

        # 2. 推理
        with torch.no_grad():
            logits = self.model(img_tensor_512)
            probs_512 = torch.sigmoid(logits)

        # 3. 还原尺寸并二值化
        probs_orig = F.interpolate(
            probs_512, 
            size=(orig_h, orig_w), 
            mode="bilinear", 
            align_corners=False
        )
        pred_mask_orig = (probs_orig > threshold).float()
        pred_np = pred_mask_orig[0, 0].cpu().numpy()
        
        return pred_np

def save_mask(pred_np, save_path):
    mask_uint8 = (pred_np * 255).astype(np.uint8)
    img = Image.fromarray(mask_uint8, mode='L')
    img.save(save_path)

def main():
    # =============== 配置 ===============
    # 请确保此处的模型是我们将要训练出来的 best 模型路径
    CHECKPOINT = os.path.join(Config.CHECKPOINT_DIR, "best_teacher_unet.pth")
    INFERENCE_ROOT = "/home/maia-user/yufeng/Data/BUS_Mix_Cls/images"
    OUTPUT_DIR = "/home/maia-user/yufeng/Data/BUS_Mix_Cls/masks_inference"
    # ====================================

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(CHECKPOINT):
        print(f"❌ 模型文件未找到: {CHECKPOINT}")
        return

    print(f"🔍 扫描文件夹: {INFERENCE_ROOT}")
    image_paths = sorted(list(Path(INFERENCE_ROOT).glob("*.png")) + list(Path(INFERENCE_ROOT).glob("*.jpg")))
    print(f"共找到 {len(image_paths)} 张图像。")

    inferencer = SimpleUnetInferencer(CHECKPOINT)
    
    print(f"💾 结果将保存至: {OUTPUT_DIR}")
    for img_path in tqdm(image_paths, desc="推理中"):
        pred_np = inferencer.run(img_path)
        save_name = img_path.name
        save_path = os.path.join(OUTPUT_DIR, save_name)
        save_mask(pred_np, save_path)
        
    print("✅ 推理全部完成！")

if __name__ == "__main__":
    main()
