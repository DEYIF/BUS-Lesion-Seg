# train.py
import torch
import torch.nn as nn
from tqdm import tqdm
import segmentation_models_pytorch as smp
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from monai.metrics.meandice import DiceMetric
from monai.transforms.post.array import AsDiscrete, Activations
from monai.transforms.compose import Compose
from monai.data.utils import decollate_batch

# 导入我们的自定义模块
from src_unet.config import Config
from src_unet.dataset import get_dataloaders
from src_unet.model import build_unet
from src_unet.utils import set_seed, save_checkpoint

def main():
    # 1. 初始化配置
    Config.setup()
    set_seed(Config.SEED)
    print(f"🚀 设备: {Config.DEVICE}, Batch Size: {Config.BATCH_SIZE}")
    
    # 2. 准备数据
    train_loader, val_loader = get_dataloaders()
    print(f"数据加载完毕: 训练集 {len(train_loader.dataset)} 张, 验证集 {len(val_loader.dataset)} 张")
    
    # 3. 准备模型
    model = build_unet().to(Config.DEVICE)
    
    # 4. 优化器与损失函数
    loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # 5. 评估指标与后处理
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_trans = Compose([
        Activations(sigmoid=True), 
        AsDiscrete(threshold=0.5)
    ])

    # 记录最佳分数
    best_metric = -1
    best_metric_epoch = -1

    # 6. 训练循环
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        model.train()
        train_loss = 0
        step = 0
        
        # --- 训练阶段 ---
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            step += 1
            images = batch["image"].to(Config.DEVICE)
            labels = batch["label"].to(Config.DEVICE)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"Train Loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- 验证阶段 (每隔 VAL_INTERVAL 次运行一次) ---
        if (epoch + 1) % Config.VAL_INTERVAL == 0:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for val_data in val_loader:
                    val_images = val_data["image"].to(Config.DEVICE)
                    val_labels = val_data["label"].to(Config.DEVICE)
                    
                    # 1. 前向传播
                    val_logits = model(val_images)
                    
                    # 2. 计算验证 Loss (仅供参考)
                    val_loss += loss_fn(val_logits, val_labels).item()
                    
                    # 3. 计算 Dice Metric
                    # MONAI 推荐使用 decollate_batch 将 batch 拆分为 list of tensors
                    # 这样处理后处理 (Post processing) 更安全
                    val_outputs = [post_trans(i) for i in decollate_batch(val_logits)]
                    val_labels = [i for i in decollate_batch(val_labels)]
                    
                    # 累积计算 Dice
                    dice_metric(y_pred=val_outputs, y=val_labels)
            
            # 聚合结果 (计算整个验证集的平均 Dice)
            avg_val_loss = val_loss / len(val_loader)
            metric = dice_metric.aggregate().item()
            dice_metric.reset() # 重置，为下一个 epoch 做准备
            
            print(f"📊 验证结果 -> Loss: {avg_val_loss:.4f} | Mean Dice: {metric:.4f}")
            
            # --- 保存最佳模型 ---
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                save_path = os.path.join(Config.CHECKPOINT_DIR, "best_teacher_unet.pth")
                
                save_checkpoint(model, epoch, best_metric, save_path)
                print(f"🌟 新纪录！模型已保存 (Dice: {best_metric:.4f})")
                
        else:
            print(f"📈 训练结束 -> Train Loss: {avg_train_loss:.4f} (本轮跳过验证)")

    print(f"\n🏁 训练完成！最佳 Dice: {best_metric:.4f} 在第 {best_metric_epoch} Epoch")

if __name__ == "__main__":
    main()