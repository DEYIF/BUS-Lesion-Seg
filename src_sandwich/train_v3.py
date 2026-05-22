"""
Sandwich V3 Training — 4-channel ablation of V2 (no UNet1 soft channel).

Ablation vs V2:
  - Student input: [RGB + soft_SAM] = 4 channels (removes soft_UNet1)
  - All other V2 improvements retained (AdamW, Cosine LR, best-val-Dice, augs).

Goal: test whether UNet1_soft channel in V2 caused shortcut learning on train set,
leading to over-reliance on UNet1 predictions at OOD test time.

Usage:
    conda run -n sam3 python src_sandwich/train_v3.py \
        --epochs 150 --batch_size 16 \
        --save_path checkpoints/sandwich_unet2_v3.pth
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src_unet.dataset import get_dataloaders_v2


def compute_dice_batch(pred_logits, gt_masks):
    """Compute mean Dice over a batch (for validation)."""
    probs = torch.sigmoid(pred_logits)
    preds = (probs > 0.5).float()
    smooth = 1e-5
    intersection = (preds * gt_masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + gt_masks.sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sandwich V2 Training")
    parser.add_argument("--epochs",     type=int,   default=150)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--val_interval", type=int, default=10,
                        help="Run validation every N epochs")
    parser.add_argument("--save_path",  type=str,
                        default="checkpoints/sandwich_unet2_v3.pth")
    parser.add_argument("--resume",     action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Model (5-channel Student UNet only) ────────────────────────────────
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=4,   # RGB + soft_SAM (no soft_UNet1)
        classes=1,
    ).to(device)

    # ── Optimizer + Schedule ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Loss ───────────────────────────────────────────────────────────────
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce_loss  = nn.BCEWithLogitsLoss()

    def criterion(pred_logits, gt_mask):
        return dice_loss(pred_logits, gt_mask) + bce_loss(pred_logits, gt_mask)

    # ── Data ───────────────────────────────────────────────────────────────
    print("Loading V2 dataloaders (soft masks)...")
    from src_unet.config import Config
    Config.BATCH_SIZE = args.batch_size
    Config.DATA_ROOT = "/home/maia-user/yufeng/Data/BUS_Mix_Seg"
    Config.EXTERNAL_TEST_DATASETS = ["BUSC", "BUID", "UDIAT", "BUSUC"]

    try:
        train_loader, val_loader = get_dataloaders_v2()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_dice = 0.0

    if args.resume and os.path.exists(args.save_path):
        print(f"Resuming from {args.save_path}")
        ckpt = torch.load(args.save_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_dice = ckpt.get("val_dice", 0.0)
        # Restore scheduler state
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed from epoch {start_epoch}, best val Dice = {best_val_dice:.4f}")

    # ── Training Loop ──────────────────────────────────────────────────────
    print(f"Training {args.epochs} epochs, val every {args.val_interval} epochs...")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            images     = batch["image"].to(device)       # (B, 3, H, W)
            masks      = batch["label"].to(device)       # (B, 1, H, W)
            sam_probs  = batch["sam_prob"].to(device)    # (B, 1, H, W)  continuous
            unet1_probs = batch["unet1_prob"].to(device) # (B, 1, H, W)  continuous

            unet2_input = torch.cat([images, sam_probs], dim=1)  # (B,4,H,W)
            unet2_logits = model(unet2_input)

            loss = criterion(unet2_logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                               "lr":   f"{scheduler.get_last_lr()[0]:.2e}"})

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)

        # ── Validation (every val_interval epochs) ─────────────────────────
        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            model.eval()
            val_dice_list = []
            with torch.no_grad():
                for batch in val_loader:
                    images      = batch["image"].to(device)
                    masks       = batch["label"].to(device)
                    sam_probs   = batch["sam_prob"].to(device)
                    unet1_probs = batch["unet1_prob"].to(device)
                    unet2_input = torch.cat([images, sam_probs], dim=1)
                    logits = model(unet2_input)
                    val_dice_list.append(compute_dice_batch(logits, masks))

            val_dice = np.mean(val_dice_list)
            print(f"Epoch {epoch+1} | train_loss={avg_loss:.4f} | val_Dice={val_dice:.4f}")

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save({
                    "epoch":               epoch,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice":            best_val_dice,
                    "train_loss":          avg_loss,
                }, args.save_path)
                print(f"  Best val Dice improved to {best_val_dice:.4f} — saved checkpoint.")
        else:
            print(f"Epoch {epoch+1} | train_loss={avg_loss:.4f}")

    print(f"\nTraining done. Best val Dice: {best_val_dice:.4f}")
    print(f"Checkpoint: {args.save_path}")


if __name__ == "__main__":
    main()
