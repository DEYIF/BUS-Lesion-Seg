import os
import sys
import torch
import numpy as np

# Adjust path so imports work
sys.path.insert(0, os.path.abspath("."))

from src_unet.inference import UnetInferencer
from src_unet.config import Config

img_path = "/home/maia-user/yufeng/Data/WHBUS/buvimgs-001/benign/benign_9161001_10/000000.png"
CHECKPOINT = os.path.join(Config.CHECKPOINT_DIR, "best_teacher_unet_allSeg.pth")

inferencer = UnetInferencer(CHECKPOINT)

item = {
    "image": img_path,
    "label": None,
    "dataset_name": "WHBUS"
}

res = inferencer.run(item)
print("Pred sum:", res['pred'].sum())
print("Pred max:", res['pred'].max())
print("Pred shape:", res['pred'].shape)

# Let's also test my batch mechanism manually without multiprocessing
from monai.transforms import Compose
temp_item = {"image": img_path}
data = inferencer.transforms_image_only(temp_item)
print("Transforms output shape:", data['image'].shape)

img_tensor_512 = data["image"].unsqueeze(0).to(inferencer.device)
with torch.no_grad():
    logits = inferencer.model(img_tensor_512)
    probs_512 = torch.sigmoid(logits)
print("Raw Model Max Prob:", probs_512.max().item())
print("Raw Model Min Prob:", probs_512.min().item())
print("Raw Model Mean Prob:", probs_512.mean().item())

