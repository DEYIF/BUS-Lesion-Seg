import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=1.0,
                 dice_weight=1.0, bce_weight=1.0):
        super().__init__()
        self.alpha = alpha # 蒸馏权重的比例
        self.T = temperature
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        # 1. 硬 Loss: Student vs Ground Truth
        self.seg_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # 2. 软 Loss: Student vs Teacher
        # MSE 简单直接，适合 Logits 回归
        self.distill_loss = nn.MSELoss() 
        # 也可以用 KLDivLoss，但要注意 Softmax/Sigmoid 处理

    def forward(self, student_logits, teacher_logits, gt_masks, confidence_mask=None):
        """
        student_logits: (B, 1, H, W) - Logits
        teacher_logits: (B, 1, H, W) - Logits (Frozen)
        gt_masks: (B, 1, H, W) - Binary (0/1)
        confidence_mask: (B, 1, H, W) - Binary mask where 1 means Teacher is confident, 0 means ambiguous.
        """
        # A. 分割任务损失 (让 SAM 学会对 - 此时对比 GT)
        # Seg Loss 始终计算
        l_seg = self.dice_weight * self.seg_loss(student_logits, gt_masks) + \
                self.bce_weight * self.bce_loss(student_logits, gt_masks)
        
        # B. 蒸馏损失 (让 SAM 学得像 Teacher - 仅在 Teacher 自信区域)
        if teacher_logits is not None:
            # 这里的 teacher_logits 不需要梯度
            with torch.no_grad():
                t_target = teacher_logits.detach()
                
            if confidence_mask is not None:
                # 只在 confidence_mask == 1 的像素上计算 MSE
                # 展平后计算可能更方便，或者用 reduction='none' 后 masking
                mse_map = F.mse_loss(student_logits, t_target, reduction='none')
                # masked_mean
                l_distill = (mse_map * confidence_mask).sum() / (confidence_mask.sum() + 1e-6)
            else:
                l_distill = self.distill_loss(student_logits, t_target)
        else:
            # Baseline Mode: No Distillation Loss
            l_distill = torch.tensor(0.0, device=student_logits.device)
        
        # 总损失
        total_loss = l_seg + self.alpha * l_distill
        
        return total_loss, l_seg, l_distill