# src_unet/config.py
import torch
import os

class Config:
    # --- 路径设置 ---
    # 请修改为你服务器上的实际数据路径
    DATA_ROOT = "/home/maia-user/yufeng/Data/BUS_Mix_Seg"
    OUTPUT_DIR = "/home/maia-user/yufeng/Data/BUS_Mix_Seg/masks_inference"
    CHECKPOINT_DIR = "/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints"
    RESUME_CHECKPOINT = "/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/latest_teacher_unet.pth" # 填入 checkpoints/latest_teacher_unet.pth 可以继续训练
    
    # --- 数据参数 ---
    IMAGE_SIZE = (512, 512)
    BATCH_SIZE = 2
    NUM_WORKERS = 2
    
    # --- 模型参数 ---
    MODEL_ARCH = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    
    # --- 训练参数 ---
    EPOCHS = 50
    VAL_INTERVAL = 5
    LEARNING_RATE = 1e-4
    SEED = 42
    # --- 环境配置 ---
    DEVICE = "cuda"            # GPU 被其他进程占满爆显存，在此强制使用 CPU 训练

    # --- 数据集选择 ---
    EXTERNAL_TEST_DATASETS = ["BUSC", "BUID", "UDIAT", "BUSUC"]  # 指定用于外部测试的数据集名称列表

    @staticmethod
    def setup():
        """确保必要的文件夹存在"""
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)