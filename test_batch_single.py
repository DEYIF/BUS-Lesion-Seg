import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Adjust path so imports work
sys.path.insert(0, os.path.abspath("."))

from src_unet.inference import UnetInferencer
from src_unet.config import Config
from src_unet.inference_video_batch import VideoFrameDataset

img_path = "/home/maia-user/yufeng/Data/WHBUS/buvimgs-001/benign/benign_9161001_10/000000.png"
CHECKPOINT = os.path.join(Config.CHECKPOINT_DIR, "best_teacher_unet_allSeg.pth")
inferencer = UnetInferencer(CHECKPOINT)

tasks = [{
    "image": img_path,
    "meta_category": "benign",
    "meta_video": "benign_9161001_10",
    "meta_filename": "000000.png"
}]

dataset = VideoFrameDataset(tasks, transform=inferencer.transforms_image_only)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

for batch in dataloader:
    images = batch["image"].to(inferencer.device)
    orig_hs = batch["orig_h"].numpy()
    orig_ws = batch["orig_w"].numpy()
    
    with torch.no_grad():
        logits = inferencer.model(images)
        probs_512 = torch.sigmoid(logits)
        
    for i in range(len(images)):
        h, w = int(orig_hs[i]), int(orig_ws[i])
        print("DataLoader orig_h, orig_w:", h, w)
        if h == 0 or w == 0:
            h, w = images[i].shape[1], images[i].shape[2]
            
        prob_single = probs_512[i:i+1]
        probs_orig = F.interpolate(
            prob_single, 
            size=(h, w), 
            mode="bilinear", 
            align_corners=False
        )
        
        pred_mask_orig = (probs_orig > 0.5).float()
        pred_np = pred_mask_orig[0, 0].cpu().numpy()
        print("Batch pred sum:", pred_np.sum())
        print("Batch pred max:", pred_np.max())
        print("Batch raw prob max:", probs_512.max().item())

