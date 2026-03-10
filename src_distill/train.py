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
    import argparse
    parser = argparse.ArgumentParser(description="SAM3 Training (Distillation vs. GT-Only)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--use_distillation", action="store_true", help="If set, uses Distillation (Teacher guiding). Otherwise uses GT-Only baseline.")
    parser.add_argument("--no_distillation", dest='use_distillation', action='store_false', help="Explicitly disable distillation (Baseline mode)")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoint if it exists.")
    parser.set_defaults(use_distillation=True) # Default is Distillation
    
    args = parser.parse_args()
    
    # 1. Setup
    unet_ckpt = "/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/best_teacher_unet_BUSC_BUID_UDIAT_BUSUC.pth"
    sam_ckpt = "/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/sam3.pt"
    
    print(f"🚀 Initializing Distillation Model... (Mode: {'Distillation' if args.use_distillation else 'GT-Only Baseline'})")
    model = DistillModel(unet_ckpt, sam_ckpt).cuda()
    
    criterion = DistillationLoss(alpha=2.0).cuda()
    
    # Optimizer: Only train student mask decoder
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-4
    )
    
    print("📂 Loading Data...")
    from src_unet.config import Config
    Config.BATCH_SIZE = args.batch_size
    train_loader, val_loader = get_dataloaders() 

    # Handle Checkpoint Paths
    save_name = "student_sam3_distilled.pth" if args.use_distillation else "student_sam3_gtonly.pth"
    save_path = f"/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/{save_name}"

    # Resume Logic
    start_epoch = 0
    if args.resume and os.path.exists(save_path):
        print(f"🔄 Resuming training from checkpoint: {save_path}")
        checkpoint = torch.load(save_path, map_location='cpu')
        
        # Check if new format (dict) or old format (state_dict only)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.student.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            print(f"✅ Resumed successfully. Starting from epoch {start_epoch + 1}")
        else:
            model.student.load_state_dict(checkpoint)
            print("✅ Loaded model weights (old format detection, starting from epoch 1)")

    # 2. Training Loop
    print(f"▶️ Starting Training for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs):
        model.train() 
        
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            images = batch['image'].cuda() # (B, 3, H, W)
            masks = batch['label'].cuda()  # (B, 1, H, W)
            
            # Generate Box Prompt from GT Mask (Training stability)
            boxes = get_box_from_mask(masks) # (B, 4)
            
            # Forward Pass
            t_logits, s_logits = model(images, boxes=boxes)
            
            # --- Loss Calculation based on Mode ---
            if args.use_distillation:
                # Distillation Mode: Use Teacher Confidence Gating
                t_probs = torch.sigmoid(t_logits)
                confidence_mask = ((t_probs > 0.9) | (t_probs < 0.1)).float()
                loss, l_seg, l_dist = criterion(s_logits, t_logits, masks, confidence_mask)
                postfix_dict = {"Loss": f"{loss.item():.4f}", "Seg": f"{l_seg.item():.4f}", "Dist": f"{l_dist.item():.4f}"}
            else:
                # GT-Only Baseline: Pure Segmentation Loss against GT
                loss, l_seg, l_dist = criterion(s_logits, None, masks, None) # criterion handles None teacher nicely for seg_loss
                loss = l_seg # Override total loss to ONLY be seg_loss
                postfix_dict = {"Loss": f"{loss.item():.4f}", "Seg": f"{l_seg.item():.4f}"}
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(postfix_dict)
            
        print(f"Epoch {epoch+1} finished. Avg Loss: {epoch_loss/len(train_loader):.4f}")
        
        # Save Model dynamically based on mode (Saved every epoch)
        save_name = "student_sam3_distilled.pth" if args.use_distillation else "student_sam3_gtonly.pth"
        save_path = f"/home/maia-user/yufeng/BUS-Lesion-Seg/checkpoints/{save_name}"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss / len(train_loader)
        }, save_path)
        print(f"💾 Model checkpoint saved to {save_path}")

if __name__ == "__main__":
    main()