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

def save_checkpoint(model, optimizer, epoch, best_metric, loss, filename):
    """保存模型及优化器状态，以便可以恢复训练"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        'loss': loss,
    }, filename)

def load_checkpoint(filename, model, optimizer=None):
    """加载模型及优化器状态（如果提供）"""
    if not os.path.exists(filename):
        print(f"⚠️ Checkpoint not found at {filename}")
        return 0, -1.0 # start_epoch, best_metric
        
    print(f"🔄 加载 Checkpoint: {filename} ...")
    checkpoint = torch.load(filename, map_location="cpu")
    
    # 兼容只保存了 model_state_dict 的旧版权重，以及只保存了权重的权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint) # try direct load

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', -1.0)
    
    return epoch, best_metric