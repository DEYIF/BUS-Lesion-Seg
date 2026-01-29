# ... import 省略 ...
from src_unet.dataset import get_dataloaders
from src_distill.model import DistillationModel
from src_distill.loss import DistillationLoss

def get_box_from_mask(mask, perturbation=0):
    """
    从 Mask (B, 1, H, W) 生成 Bounding Box (B, 4)
    格式: [x1, y1, x2, y2]
    """
    # 这是一个简化版，实际需要处理 batch 循环
    # 这里建议使用 monai 或 torchvision 的 ops
    # 为了演示原理，写个伪代码逻辑：
    boxes = []
    for m in mask:
        y_indices, x_indices = torch.where(m[0] > 0.5)
        if len(y_indices) == 0:
            # 空 mask，给个全图或 dummy box
            boxes.append([0, 0, 1, 1]) 
        else:
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            # 加扰动 noise ...
            boxes.append([x_min, y_min, x_max, y_max])
    
    return torch.tensor(boxes).float().cuda()

def main():
    # 1. 准备
    unet_ckpt = "checkpoints/best_teacher_unet.pth"
    sam_ckpt = "checkpoints/sam_vit_b_01ec64.pth" # 需下载
    
    model = DistillationModel(unet_ckpt, sam_ckpt).cuda()
    criterion = DistillationLoss(alpha=2.0) # 调大 alpha 强调蒸馏
    
    # 这里的参数只包含 SAM 的 Decoder 部分 (如果冻结了 Encoder)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-4
    )
    
    train_loader, _ = get_dataloaders() # 复用之前的 loader

    # 2. 循环
    for epoch in range(50):
        model.train() # Teacher 内部已经强制 eval 了，这里只会开启 Student 的 train
        
        for batch in tqdm(train_loader):
            images = batch['image'].cuda()
            masks = batch['label'].cuda()
            
            # 关键：动态生成 Prompt Box
            # SAM 需要 prompt 才能工作得好
            with torch.no_grad():
                # 为了简单，这里需要你实现一个从 masks 生成 bounding box 的函数
                # SAM 要求的格式是 (B, N, 4)
                boxes = get_box_from_mask(masks) 
                # SAM 需要 transform box 到 1024 尺寸，这里需要注意 coordinates 转换
                # 如果你用 monai 的 SAM wrapper 会自动处理，这里假设是手动
            
            # 前向
            t_logits, s_logits = model(images, boxes=boxes)
            
            # 计算 Loss
            loss, l_seg, l_dist = criterion(s_logits, t_logits, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log...