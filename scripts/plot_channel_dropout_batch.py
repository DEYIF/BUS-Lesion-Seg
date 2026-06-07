"""
Sandwich V2 — Channel dropout 训练时 batch 示意图。

可视化一个 batch=4 的 Student 输入，5 个通道并排（R/G/B/SAM³ soft/UNet1 soft）。
被 dropout 命中的样本，UNet1 列画灰色 + 红 ✗。
"""
import os
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
font_manager.fontManager.addfont(
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
mpl.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

DATA_ROOT = Path("/home/maia-user/yufeng/Data/BUS_Mix_Seg")
OUT_PATH  = Path("/home/maia-user/yufeng/BUS-Lesion-Seg/Results/"
                 "sandwich_v2_optimizations/channel_dropout_batch.png")

# 4 个 batch 样本；True 表示该样本本次 UNet1 通道被 dropout 命中
BATCH = [
    ("BUID_122_Malignant.png", False),
    ("BUS-UCLM_ASSC_018.png",  True),    # ← dropout 命中
    ("BUID_8_Benign.png",      False),
    ("BUSUC_00402.png",        True),    # ← dropout 命中
]

CHANNEL_TITLES = ["R", "G", "B", "SAM³ soft", "UNet1 soft"]


def load_inputs(name: str):
    img = np.array(Image.open(DATA_ROOT / "images" / name).convert("RGB"))
    sam = np.array(Image.open(DATA_ROOT / "sam3_probs_lora" / name).convert("L"),
                   dtype=np.float32) / 255.0
    une = np.array(Image.open(DATA_ROOT / "unet1_probs"    / name).convert("L"),
                   dtype=np.float32) / 255.0
    return img, sam, une


def render():
    n_rows = len(BATCH)
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.3))

    for r, (name, dropped) in enumerate(BATCH):
        img, sam, une = load_inputs(name)
        rgb = [(img[..., 0], "Reds"),
               (img[..., 1], "Greens"),
               (img[..., 2], "Blues")]
        for c, (ch, cmap) in enumerate(rgb):
            ax = axes[r, c]
            ax.imshow(ch, cmap=cmap, vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            if r == 0:
                ax.set_title(CHANNEL_TITLES[c], fontsize=12)

        ax_sam = axes[r, 3]
        ax_sam.imshow(sam, cmap="jet", vmin=0, vmax=1)
        ax_sam.set_xticks([]); ax_sam.set_yticks([])
        for s in ax_sam.spines.values(): s.set_visible(False)
        if r == 0:
            ax_sam.set_title(CHANNEL_TITLES[3], fontsize=12)

        ax_u1 = axes[r, 4]
        if dropped:
            ax_u1.imshow(np.zeros_like(une), cmap="gray", vmin=0, vmax=1)
            ax_u1.text(0.5, 0.5, "✗", color="red", fontsize=60, fontweight="bold",
                       ha="center", va="center", transform=ax_u1.transAxes)
            for s in ax_u1.spines.values():
                s.set_visible(True); s.set_color("red"); s.set_linewidth(2.5)
        else:
            ax_u1.imshow(une, cmap="jet", vmin=0, vmax=1)
            for s in ax_u1.spines.values(): s.set_visible(False)
        ax_u1.set_xticks([]); ax_u1.set_yticks([])
        if r == 0:
            ax_u1.set_title(CHANNEL_TITLES[4], fontsize=12)

        label = f"sample {r+1}"
        if dropped:
            label += "  (UNet1 dropped)"
            color = "red"
        else:
            color = "black"
        axes[r, 0].set_ylabel(label, fontsize=11, rotation=0, ha="right",
                              va="center", color=color, labelpad=15)

    fig.suptitle("Sandwich V2 — Per-sample channel dropout on UNet1 (p = 0.3)\n"
                 "训练时随机静音整张 UNet1 通道，防止 Student 把 UNet1 当作答案抄",
                 fontsize=13)
    fig.tight_layout(rect=(0.04, 0, 1, 0.92))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    render()
