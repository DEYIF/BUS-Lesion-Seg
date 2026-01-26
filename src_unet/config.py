# src_unet/config.py
import torch
import os

class Config:
    # --- 路径设置 ---
    # 请修改为你服务器上的实际数据路径
    DATA_ROOT = "/home/maia-user/yufeng/Data/BUS_Mix_Cls"
    OUTPUT_DIR = "/home/maia-user/yufeng/Data/BUS_Mix_Cls/masks_inference"
    CHECKPOINT_DIR = "/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints"
    
    # --- 数据参数 ---
    IMAGE_SIZE = (512, 512)
    BATCH_SIZE = 8  # 根据显存调整 (3090/4090 可开 16-32)
    NUM_WORKERS = 4
    
    # --- 模型参数 ---
    MODEL_ARCH = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    
    # --- 训练参数 ---
    EPOCHS = 50
    VAL_INTERVAL = 10
    LEARNING_RATE = 1e-4
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 数据集选择 ---
    EXTERNAL_TEST_DATASETS = ["Thammasat"]  # 指定用于外部测试的数据集名称列表

    @staticmethod
    def setup():
        """确保必要的文件夹存在"""
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)