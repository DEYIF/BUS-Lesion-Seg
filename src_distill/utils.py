import torch
import numpy as np

def get_boxes_from_mask(masks, noise_scale=0.1):
    """
    从 Mask 生成 Bounding Box，并加入随机扰动。
    masks: (B, 1, H, W) 0/1 张量
    return: (B, 1, 4) 格式为 [x1, y1, x2, y2] 的 boxes
    """
    B, _, H, W = masks.shape
    boxes = []

    device = masks.device

    for i in range(B):
        mask = masks[i, 0]
        # 获取非零元素的索引
        y_indices, x_indices = torch.where(mask > 0.5)

        if len(y_indices) == 0:
            # 如果是空 Mask（没有病灶），给一个全图的 Box 或者 Dummy Box
            # SAM 对空 Prompt 的处理比较敏感，这里给一个极小的 box 放在角落
            box = [0, 0, 1, 1] 
        else:
            x_min, x_max = x_indices.min().item(), x_indices.max().item()
            y_min, y_max = y_indices.min().item(), y_indices.max().item()
            
            # === 添加随机扰动 (Noise) ===
            # 模拟用户点击不准的情况，增加鲁棒性
            w_box = x_max - x_min
            h_box = y_max - y_min
            
            noise_x = w_box * noise_scale
            noise_y = h_box * noise_scale
            
            x_min = max(0, x_min - np.random.uniform(0, noise_x))
            y_min = max(0, y_min - np.random.uniform(0, noise_y))
            x_max = min(W, x_max + np.random.uniform(0, noise_x))
            y_max = min(H, y_max + np.random.uniform(0, noise_y))
            
            box = [x_min, y_min, x_max, y_max]

        boxes.append(box)

    # SAM 2 期望的输入格式通常是 (B, N_prompts, 4)
    # 这里每张图 1 个 box
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=device).unsqueeze(1)
    return boxes_tensor