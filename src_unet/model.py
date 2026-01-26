# src_unet/model.py
import segmentation_models_pytorch as smp
from .config import Config

def build_unet():
    """创建一个 U-Net 实例"""
    model = smp.Unet(
        encoder_name=Config.MODEL_ARCH,
        encoder_weights=Config.ENCODER_WEIGHTS,
        in_channels=3, # 即使是超声，我们通常也使用 RGB 输入以利用预训练权重
        classes=1,
    )
    return model