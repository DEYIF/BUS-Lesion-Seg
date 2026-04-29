"""
Compose A1 fallback (over-segmentation rule R6) on top of an existing V2 mask
output directory, without re-running inference.

Reads:
  Experiments/<v2_dir>/<dataset>/masks/<filename>.png  — V2 binary masks
  Experiments/<v1_dir>/<dataset>/masks/<filename>.png  — V1+LoRA binary masks
  Data/BUS_Mix_Seg/labels/<filename>.png               — GT (for new metrics)

Writes:
  Experiments/<out_dir>/<dataset>/masks/<filename>.png  — composed masks
  Experiments/<out_dir>/metrics.csv                     — recomputed Dice/IoU
  Experiments/<out_dir>/statistics.csv                  — per-dataset summary

Trigger (rule R6): v2_area > over_ratio * v1_area AND v1_area > min_area * image_size

Usage:
  conda run -n sam3 python BUS-Lesion-Seg/scripts/compose_v2_a1.py \\
      --v2_dir sandwich_v2_lora_drop \\
      --out_dir sandwich_v2_lora_drop_a1
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path("/home/maia-user/yufeng/BUS-Lesion-Seg")
EXP  = ROOT / "Experiments"
DATA = Path("/home/maia-user/yufeng/Data/BUS_Mix_Seg")
EPS  = 1e-5


def load_mask(path, target_shape=None):
    arr = np.array(Image.open(path).convert("L"))
    if target_shape and arr.shape != target_shape:
        arr = np.array(
            Image.fromarray(arr).resize((target_shape[1], target_shape[0]), Image.NEAREST)
        )
    return (arr > 127).astype(np.uint8)


def dice(p, g):
    inter = (p * g).sum()
    return (2.0 * inter + EPS) / (p.sum() + g.sum() + EPS)


def iou(p, g):
    inter = (p * g).sum()
    union = p.sum() + g.sum() - inter
    return (inter + EPS) / (union + EPS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2_dir", required=True)
    parser.add_argument("--v1_dir", default="sandwich_v1_lora")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--over_ratio", type=float, default=1.5)
    parser.add_argument("--v1_min_area", type=float, default=0.01)
    args = parser.parse_args()

    v2_root  = EXP / args.v2_dir
    v1_root  = EXP / args.v1_dir
    out_root = EXP / args.out_dir
    if not v2_root.exists():
        print(f"V2 dir not found: {v2_root}")
        sys.exit(1)
    if not v1_root.exists():
        print(f"V1 dir not found: {v1_root}")
        sys.exit(1)
    out_root.mkdir(parents=True, exist_ok=True)

    v2_metrics = pd.read_csv(v2_root / "metrics.csv")  # Dataset, Filename, Dice, IoU

    rows = []
    a1_count = 0
    print(f"Composing A1 (over_ratio={args.over_ratio}, v1_min_area={args.v1_min_area})...")
    for _, r in tqdm(v2_metrics.iterrows(), total=len(v2_metrics)):
        ds  = r["Dataset"]
        fn  = r["Filename"]
        v2_path = v2_root / ds / "masks" / fn
        v1_path = v1_root / ds / "masks" / fn
        gt_path = DATA / "labels" / fn
        if not v2_path.exists() or not v1_path.exists() or not gt_path.exists():
            continue

        v2_mask = load_mask(v2_path)
        v1_mask = load_mask(v1_path, target_shape=v2_mask.shape)

        v1_area = float(v1_mask.sum())
        v2_area = float(v2_mask.sum())
        img_sz  = float(v2_mask.size)

        used_a1 = False
        if v1_area > args.v1_min_area * img_sz and v2_area > args.over_ratio * v1_area:
            final = v1_mask
            used_a1 = True
            a1_count += 1
        else:
            final = v2_mask

        # Write final mask
        out_mask_dir = out_root / ds / "masks"
        out_mask_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray((final * 255).astype(np.uint8), mode="L").save(out_mask_dir / fn)

        # Recompute metrics
        gt_mask = load_mask(gt_path, target_shape=final.shape)
        d = dice(final.astype(np.float32), gt_mask.astype(np.float32))
        i = iou(final.astype(np.float32), gt_mask.astype(np.float32))
        rows.append({"Dataset": ds, "Filename": fn, "Dice": d, "IoU": i})

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "metrics.csv", index=False)
    print(f"\nA1 fired: {a1_count} / {len(rows)}")
    print(f"Overall mean Dice: {df['Dice'].mean():.4f}  IoU: {df['IoU'].mean():.4f}")
    print(df.groupby("Dataset")[["Dice", "IoU"]].mean())

    try:
        sys.path.insert(0, str(ROOT))
        from seg_utils.calculate_stats import calculate_stats_from_csv
        calculate_stats_from_csv(str(out_root / "metrics.csv"),
                                  str(out_root / "statistics.csv"))
        print(f"Wrote statistics.csv")
    except Exception as e:
        print(f"(skip statistics.csv: {e})")


if __name__ == "__main__":
    main()
