"""
Sandwich V2 — 累加式消融柱状图。

读取 5 套实验的 metrics.csv，按 4 个数据集 + Overall 画并排柱：
    V1  →  +5ch+soft  →  +channel dropout  →  +fallback (full V2)
"""
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
font_manager.fontManager.addfont(
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
mpl.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP_ROOT = Path("/home/maia-user/yufeng/BUS-Lesion-Seg/Experiments")
OUT_PATH = Path("/home/maia-user/yufeng/BUS-Lesion-Seg/Results/"
                "sandwich_v2_optimizations/v2_ablation_bars.png")

# (label, experiment-dir, color)
CONFIGS = [
    ("V1 baseline",         "sandwich_v1_lora",        "#cfd8dc"),
    ("+ 5ch + soft prob",   "sandwich_v2_lora",        "#90caf9"),
    ("+ channel dropout",   "sandwich_v2_lora_drop",   "#42a5f5"),
    ("+ fallback (full V2)", "sandwich_v2_lora_drop_a1","#1565c0"),
]

DATASETS = ["BUSC", "BUID", "UDIAT", "BUSUC"]


def load_means():
    rows = []
    for label, d, _ in CONFIGS:
        df = pd.read_csv(EXP_ROOT / d / "metrics.csv")
        per_ds = df.groupby("Dataset")["Dice"].mean()
        overall = df["Dice"].mean()
        rows.append({"config": label,
                     **{ds: per_ds.get(ds, np.nan) for ds in DATASETS},
                     "Overall": overall})
    return pd.DataFrame(rows).set_index("config")


def render():
    means = load_means()
    groups = DATASETS + ["Overall"]
    n_groups  = len(groups)
    n_configs = len(CONFIGS)
    bar_w = 0.18
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, (label, _, color) in enumerate(CONFIGS):
        vals = [means.loc[label, g] for g in groups]
        offsets = x + (i - (n_configs - 1) / 2) * bar_w
        bars = ax.bar(offsets, vals, bar_w, label=label, color=color,
                      edgecolor="white", linewidth=0.5)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.axvline(n_groups - 1.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel("Mean Dice", fontsize=11)
    ax.set_ylim(0.78, 0.97)
    ax.set_title("Sandwich V2 — 累加消融：每加一项优化的 Dice 增益",
                 fontsize=13)
    ax.legend(loc="lower right", fontsize=9.5, ncol=2, framealpha=0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")
    print()
    print("Per-config Dice means:")
    print(means.round(4).to_string())


if __name__ == "__main__":
    render()
