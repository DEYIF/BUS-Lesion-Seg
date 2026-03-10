import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath("."))
from src_unet.inference import UnetInferencer
from src_unet.config import Config
from src_unet.inference_video_batch import VideoFrameDataset

img_path = "/home/maia-user/yufeng/Data/WHBUS/buvimgs-001/benign/benign_9161001_10/000000.png"
CHECKPOINT = os.path.join(Config.CHECKPOINT_DIR, "best_teacher_unet_allSeg.pth")
inferencer = UnetInferencer(CHECKPOINT)

# Create a batch of the same image just to test workers
tasks = [{
    "image": img_path,
    "meta_category": "benign",
    "meta_video": "benign_9161001_10",
    "meta_filename": f"00000{i}.png"
} for i in range(8)]

dataset = VideoFrameDataset(tasks, transform=inferencer.transforms_image_only)
dataloader = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=False)

for batch in dataloader:
    images = batch["image"]
    print("Batch Images shape:", images.shape)
    print("Batch Images max:", images.max().item())
    print("Batch Images min:", images.min().item())
    print("Batch Images mean:", images.mean().item())
    
    # We can also check if the model gives good probs
    images = images.to(inferencer.device)
    with torch.no_grad():
        logits = inferencer.model(images)
        probs_512 = torch.sigmoid(logits)
    print("Probs max:", probs_512.max().item())
    print("Probs sum:", (probs_512 > 0.5).float().sum().item())
    break
