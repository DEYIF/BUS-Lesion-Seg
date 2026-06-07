"""
Sandwich V2 — Hard mask vs Soft probability 对比图。

每张图 = 一个样本的 2x3 面板：
    Row 0 (SAM3)  : Input(+GT)  | SAM3  Hard mask | SAM3  Soft prob (heatmap)
    Row 1 (UNet1) : Input(+GT)  | UNet1 Hard mask | UNet1 Soft prob (heatmap)

数据直接读 precompute_masks_v2.py 写出的 0–255 uint8 概率 PNG，hard mask 通过
阈值 0.5 在线计算得到。无需 GPU、无需加载模型。

Usage:
    python BUS-Lesion-Seg/scripts/plot_hard_vs_soft.py \
        --samples BUID_122_Malignant.png BUID_8_Benign.png \
        --out_dir BUS-Lesion-Seg/Results/sandwich_v2_hard_vs_soft
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_prob_png(path: Path) -> np.ndarray:
    """读取 precompute_masks_v2 写出的 uint8 PNG，返回 [0,1] float32。"""
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return arr


def load_image_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_label_binary(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 127).astype(np.uint8)


def render_one(sample: str, data_root: Path, sam_prob_dir: str,
               out_path: Path, threshold: float = 0.5) -> None:
    img      = load_image_rgb(data_root / "images" / sample)
    label    = load_label_binary(data_root / "labels" / sample)
    sam_soft = load_prob_png(data_root / sam_prob_dir / sample)
    une_soft = load_prob_png(data_root / "unet1_probs" / sample)

    sam_hard = (sam_soft > threshold).astype(np.uint8)
    une_hard = (une_soft > threshold).astype(np.uint8)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    row_titles = ["SAM3", "UNet1"]
    softs      = [sam_soft, une_soft]
    hards      = [sam_hard, une_hard]

    for row in range(2):
        ax_in, ax_hard, ax_soft = axes[row]

        ax_in.imshow(img)
        ax_in.contour(label, levels=[0.5], colors="lime", linewidths=1.5)
        ax_in.set_title(f"{row_titles[row]} — Input + GT")
        ax_in.axis("off")

        ax_hard.imshow(hards[row], cmap="gray", vmin=0, vmax=1)
        ax_hard.set_title(f"{row_titles[row]} Hard mask (thr={threshold})")
        ax_hard.axis("off")

        im = ax_soft.imshow(softs[row], cmap="jet", vmin=0, vmax=1)
        ax_soft.set_title(f"{row_titles[row]} Soft probability")
        ax_soft.axis("off")
        plt.colorbar(im, ax=ax_soft, fraction=0.046, pad=0.04)

    fig.suptitle(f"Sandwich V2 — Hard mask vs Soft probability\n{sample}",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default="/home/maia-user/yufeng/Data/BUS_Mix_Seg")
    parser.add_argument("--sam_prob_dir", type=str, default="sam3_probs_lora",
                        choices=["sam3_probs", "sam3_probs_lora"],
                        help="SAM3 概率图目录（V2 默认 LoRA 版本）。")
    parser.add_argument("--samples", nargs="+",
                        default=["BUID_122_Malignant.png", "BUID_8_Benign.png"],
                        help="要可视化的文件名（需同时存在于 images/labels/"
                             "<sam_prob_dir>/unet1_probs 四个目录）。")
    parser.add_argument("--out_dir", type=str,
                        default="/home/maia-user/yufeng/BUS-Lesion-Seg/Results/"
                                "sandwich_v2_hard_vs_soft")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir   = Path(args.out_dir)

    for sample in args.samples:
        stem = os.path.splitext(sample)[0]
        out_path = out_dir / f"hard_vs_soft_{stem}.png"
        render_one(sample, data_root, args.sam_prob_dir, out_path,
                   threshold=args.threshold)


if __name__ == "__main__":
    main()
