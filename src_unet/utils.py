# src_unet/utils.py
import torch
import numpy as np
import random
import os

def set_seed(seed):
    """固定随机种子，保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def save_checkpoint(model, epoch, loss, filename):
    """保存模型"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, filename)