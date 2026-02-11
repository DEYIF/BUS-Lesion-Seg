# src_distill/config.py
import torch
import os

class Config:
    # --- 路径设置 ---
    UNET_CKPT = "/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/best_teacher_unet_BUSC_BUID_UDIAT_BUSUC.pth"
    SAM_CKPT = "/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/sam3.pt"
    DATA_ROOT = "/home/maia-user/yufeng/Data/BUS_Mix_Seg"
    OUTPUT_DIR = "/home/maia-user/yufeng/Data/BUS_Mix_Seg/masks_inference"
    
    # --- 数据参数 ---
    IMAGE_SIZE = (512, 512)
    BATCH_SIZE = 8  # 根据显存调整 (3090/4090 可开 16-32)
    NUM_WORKERS = 4
    
    # --- 模型参数 ---
    MODEL_ARCH = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    
    # --- 训练参数 ---
    EPOCHS = 50
    VAL_INTERVAL = 5
    LEARNING_RATE = 1e-4
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 数据集选择 ---
    EXTERNAL_TEST_DATASETS = ["BUSC", "BUID", "UDIAT", "BUSUC"]  # 指定用于外部测试的数据集名称列表

    @staticmethod
    def setup():
        """确保必要的文件夹存在"""
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)