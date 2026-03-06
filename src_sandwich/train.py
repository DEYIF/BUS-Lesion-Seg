import sys
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import segmentation_models_pytorch as smp

# Add parent directory to path to allow imports
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src_unet.dataset import get_dataloaders

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Optimized Sandwich Training (U-Net2 Only)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--save_path", type=str, default="checkpoints/sandwich_unet2.pth", help="Checkpoint save path")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Initializing U-Net 2 (Stage 3 Refiner) on {device}...")
    
    # We only need the final U-Net now!
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=4, # 3 (RGB) + 1 (SAM Mask)
        classes=1,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Loss Function (Dice + BCE)
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce_loss = nn.BCEWithLogitsLoss()
    
    def criterion(pred_logits, gt_mask):
        return dice_loss(pred_logits, gt_mask) + bce_loss(pred_logits, gt_mask)

    # Data Loaders WITH PRECOMPUTED MASKS
    print("📂 Loading Data with precomputed SAM3 masks...")
    
    # Ensure config knows about batch size if it uses it directly (or we can override it if dataset.py was modified, but let's stick to default Config usage built-in)
    from src_unet.config import Config
    Config.BATCH_SIZE = args.batch_size 
    
    try:
        train_loader, val_loader = get_dataloaders(use_sam_mask=True)
    except Exception as e:
         print(f"❌ DataLoader error: {e}")
         print("Did you run `src_sandwich/precompute_masks.py` first?")
         return
    
    best_loss = float('inf')
    
    print(f"▶️ Starting Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            images = batch['image'].to(device)   # (B, 3, H, W)
            masks = batch['label'].to(device)    # (B, 1, H, W)
            
            # Crucial: The precomputed SAM masks
            if 'sam_mask' not in batch:
                raise ValueError("sam_mask not found in batch! Precomputation missing?")
                
            sam_masks = batch['sam_mask'].to(device) # (B, 1, H, W)
            
            # Concat Image and SAM Mask
            unet2_input = torch.cat([images, sam_masks], dim=1) # (B, 4, H, W)
            
            # Forward Pass (Only U-Net2)
            unet2_logits = model(unet2_input)
            
            # Loss
            loss = criterion(unet2_logits, masks)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg Train Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"🌟 Best loss improved to {best_loss:.4f}, saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, args.save_path)
            
    print(f"💾 Training complete. Best weights at {args.save_path}")

if __name__ == "__main__":
    main()
