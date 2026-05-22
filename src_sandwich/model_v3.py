"""
Sandwich Model V3 — 4-channel Student UNet (Option A ablation).

Ablation vs model_v2.py:
  - Removes the UNet1_soft input channel.
  - Student sees only [RGB (3ch) + SAM_soft (1ch)] = 4 channels.
  - Hypothesis: 5-channel V2 suffered from shortcut learning where Student
    over-relied on UNet1_soft (perfect on train set, wrong on OOD test sets).
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from src_sandwich.model_v2 import SandwichModelV2


class SandwichModelV3(SandwichModelV2):
    def __init__(self, unet1_ckpt, sam_ckpt, unet2_arch="resnet34", unet2_weights="imagenet"):
        super().__init__(unet1_ckpt, sam_ckpt, unet2_arch, unet2_weights)

        print(f"V3 Layer 3: Rebuilding U-Net Student ({unet2_arch}, in_channels=4)...")
        self.unet2 = smp.Unet(
            encoder_name=unet2_arch,
            encoder_weights=unet2_weights,
            in_channels=4,
            classes=1,
        )

    def forward(self, images):
        if hasattr(images, "as_tensor"):
            images = images.as_tensor()

        with torch.no_grad():
            unet1_logits = self.unet1(images)
            unet1_soft   = torch.sigmoid(unet1_logits)
            unet1_mask   = (unet1_soft > 0.5).float()
            boxes        = self.get_box_from_mask(unet1_mask)

        with torch.no_grad():
            sam_logits = self.forward_sam(images, boxes)
            sam_soft   = torch.sigmoid(sam_logits).detach()

        unet2_input  = torch.cat([images, sam_soft], dim=1)
        unet2_logits = self.unet2(unet2_input)

        return unet2_logits, sam_soft, unet1_soft, unet1_mask
