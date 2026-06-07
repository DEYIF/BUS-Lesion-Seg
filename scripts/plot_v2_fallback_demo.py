"""
Sandwich V2 — Inference fallback 触发示意图（仅图片 + overlay）。

两行 A1 触发案例 × 4 列：Input+GT / V2 raw / V1 fallback / Final
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATA_ROOT = Path("/home/maia-user/yufeng/Data/BUS_Mix_Seg")
EXP_ROOT  = Path("/home/maia-user/yufeng/BUS-Lesion-Seg/Experiments")
OUT_PATH  = Path("/home/maia-user/yufeng/BUS-Lesion-Seg/Results/"
                 "sandwich_v2_optimizations/fallback_demo.png")

CASES = [
    ("BUSUC", "BUSUC_0022.png"),
    ("UDIAT", "UDIAT_000099.png"),
]

V2_RAW_DIR = "sandwich_v2_lora_drop"
V1_DIR     = "sandwich_v1_lora"
FINAL_DIR  = "sandwich_v2_lora_drop_a1"


def load_rgb(p):  return np.array(Image.open(p).convert("RGB"))
def load_mask(p): return (np.array(Image.open(p).convert("L")) > 127).astype(np.uint8)


def panel(ax, img, mask=None, color=(1, 0, 0), gt=None):
    ax.imshow(img)
    if mask is not None:
        rgba = np.zeros((*mask.shape, 4))
        rgba[..., 0:3] = color
        rgba[..., 3]   = mask * 0.45
        ax.imshow(rgba)
    if gt is not None:
        ax.contour(gt, levels=[0.5], colors="lime", linewidths=1.8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


def render():
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))

    for r, (ds, fn) in enumerate(CASES):
        img      = load_rgb(DATA_ROOT / "images" / fn)
        gt       = load_mask(DATA_ROOT / "labels" / fn)
        v2_raw   = load_mask(EXP_ROOT / V2_RAW_DIR / ds / "masks" / fn)
        v1_mask  = load_mask(EXP_ROOT / V1_DIR     / ds / "masks" / fn)
        final    = load_mask(EXP_ROOT / FINAL_DIR  / ds / "masks" / fn)

        panel(axes[r, 0], img, gt=gt)
        panel(axes[r, 1], img, mask=v2_raw,  color=(1, 0, 0))
        panel(axes[r, 2], img, mask=v1_mask, color=(0.1, 0.45, 1))
        panel(axes[r, 3], img, mask=final,   color=(1, 0, 0), gt=gt)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.02)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    render()
