import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.ops import masks_to_boxes
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src_unet.dataset import get_dataloaders
from src_distill.model import DistillModel
from src_distill.loss import DistillationLoss

def get_box_from_mask(mask, perturbation=0):
    """
    Generate Bounding Box (B, 4) from Mask (B, 1, H, W)
    Format: [x1, y1, x2, y2]
    """
    # mask: (B, 1, H, W)
    # Binary mask
    B, _, H, W = mask.shape
    boxes_list = []
    
    # We iterate because masks_to_boxes expects (N, H, W) binary masks
    # where N is number of objects. Here we have B images, each with 1 object (Foreground).
    # If a mask is empty, masks_to_boxes might fail or return empty.
    
    for i in range(B):
        m = mask[i, 0] # (H, W)
        if m.sum() == 0:
            # Empty mask, return a dummy box or full image?
            # SAM needs a prompt. Use full image center point or full box?
            # Let's use [0, 0, 1, 1] or [0, 0, W, H]
            box = torch.tensor([0, 0, W, H], device=mask.device, dtype=torch.float32)
        else:
            # torchvision masks_to_boxes expects (N, H, W), here N=1
            box = masks_to_boxes(m.unsqueeze(0))[0] # (4,)
            
            # Add perturbation if needed (augmentation during training)
            if perturbation > 0:
                noise = torch.randn_like(box) * perturbation
                box = box + noise
                # Clamp
                box[0::2] = box[0::2].clamp(0, W)
                box[1::2] = box[1::2].clamp(0, H)
        
        boxes_list.append(box)
    
    return torch.stack(boxes_list) # (B, 4)

def main():
    # 1. Setup
    # Paths from user environment
    unet_ckpt = "/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/best_teacher_unet_BUSC_BUID_UDIAT_BUSUC.pth"
    sam_ckpt = "/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/sam3.pt"
    
    print("🚀 Initializing Distillation Model...")
    # NOTE: DistillModel import was changed to DistillModel in model.py, check imports!
    # In my previous edit I changed class name to DistillModel (was DistillationModel in train.py before)
    model = DistillModel(unet_ckpt, sam_ckpt).cuda()
    
    criterion = DistillationLoss(alpha=2.0).cuda()
    
    # Optimizer: Only train student mask decoder
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-4
    )
    
    print("📂 Loading Data...")
    train_loader, val_loader = get_dataloaders() 

    # 2. Training Loop
    epochs = 1
    print(f"▶️ Starting Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train() # This enables student training layers
        
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            images = batch['image'].cuda() # (B, 3, H, W)
            masks = batch['label'].cuda()  # (B, 1, H, W)
            
            # --- 1. Teacher Inference (for Distillation Targets & Box Prompt) ---
            # We need Teacher Logits for Distillation Loss
            # We ALSO need Teacher Prediction (Mask) to generate Box Prompt if "Prompt-Free" intent.
            # But the user said: "Use GT Box for training stability, but try Unet Box for inference".
            # The User also suggested: "Input X -> Frozen U-Net -> P_unet. Input X -> SAM (Prompted by P_unet or GT)".
            # "Usually use GT Box for training stability."
            # Let's use GT Box for training today as per standard practice, 
            # OR mix them? User said "Prompt generation (Key point): Usually GT. Advanced: use P_unet".
            # Let's implement BOTH: Randomly switch or just use GT for now to ensure convergence.
            
            # Generate Box Prompt from GT Mask (Training stability)
            boxes = get_box_from_mask(masks) # (B, 4)
            
            # --- 2. Forward Pass ---
            # model(images, boxes) returns teacher_logits, student_logits
            # teacher_logits comes from running Teacher(images) inside model.forward
            t_logits, s_logits = model(images, boxes=boxes)
            
            # --- 3. Confidence Gating for Loss ---
            # t_logits into probabilities
            t_probs = torch.sigmoid(t_logits)
            
            # Confidence Mask: 1 if p > 0.9 or p < 0.1
            # "High confidence"
            confidence_mask = ((t_probs > 0.9) | (t_probs < 0.1)).float()
            
            # --- 4. Loss & Backprop ---
            loss, l_seg, l_dist = criterion(s_logits, t_logits, masks, confidence_mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Seg": f"{l_seg.item():.4f}", "Dist": f"{l_dist.item():.4f}"})
            
        print(f"Epoch {epoch+1} finished. Avg Loss: {epoch_loss/len(train_loader):.4f}")
        
        # Validation could be added here
        
    # Save Model
    torch.save(model.student.state_dict(), "/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/student_sam3_distilled.pth")
    print("💾 Model saved to /home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/student_sam3_distilled.pth")

if __name__ == "__main__":
    main()