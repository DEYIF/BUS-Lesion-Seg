"""
Paired statistical significance tests between models on the same 1456 external
test samples.

Tests run per pair:
- Wilcoxon signed-rank test (non-parametric, paired) — primary
- Paired t-test (parametric, paired) — companion
- Bootstrap 95% CI of mean Dice difference (10000 resamples)
- Effect size: Cliff's delta + median difference

Reports overall and per-dataset.
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/home/maia-user/yufeng/BUS-Lesion-Seg")
EXP  = ROOT / "Experiments"


def load(name):
    df = pd.read_csv(EXP / name / "metrics.csv")
    return df.set_index("Filename")[["Dataset", "Dice"]]


def cliffs_delta(x, y):
    """Cliff's delta: P(X > Y) - P(X < Y) on paired samples."""
    nx, ny = len(x), len(y)
    pos = sum(1 for a, b in zip(x, y) if a > b)
    neg = sum(1 for a, b in zip(x, y) if a < b)
    return (pos - neg) / nx


def bootstrap_ci_diff(x, y, n_boot=10000, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    diffs = x - y
    n = len(diffs)
    boot = rng.choice(diffs, size=(n_boot, n), replace=True)
    means = boot.mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return diffs.mean(), lo, hi


def compare(name_a, df_a, name_b, df_b, dataset_filter=None):
    common = df_a.index.intersection(df_b.index)
    if dataset_filter is not None:
        a_ds = df_a.loc[common, "Dataset"]
        common = common[a_ds == dataset_filter]

    a = df_a.loc[common, "Dice"].astype(float).to_numpy()
    b = df_b.loc[common, "Dice"].astype(float).to_numpy()

    if len(a) < 5:
        return f"  ({dataset_filter or 'Overall'}: n={len(a)}, skip)"

    diff = a - b
    nz_diff = diff[diff != 0]   # Wilcoxon needs non-zero diffs

    if len(nz_diff) >= 1:
        try:
            w_stat, w_p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
    else:
        w_stat, w_p = float("nan"), float("nan")

    t_stat, t_p = stats.ttest_rel(a, b)
    median_diff = float(np.median(diff))
    mean_a, mean_b = float(a.mean()), float(b.mean())
    cd = cliffs_delta(a, b)
    mean_diff, ci_lo, ci_hi = bootstrap_ci_diff(a, b)

    sig = ""
    if w_p < 0.001:   sig = "***"
    elif w_p < 0.01:  sig = "**"
    elif w_p < 0.05:  sig = "*"
    elif w_p < 0.1:   sig = "."
    else:             sig = "ns"

    label = dataset_filter or "Overall"
    return (f"  {label:8s} (n={len(a):4d}): "
            f"mean {name_a}={mean_a:.4f} {name_b}={mean_b:.4f} "
            f"Δ={mean_diff:+.4f} [95% CI {ci_lo:+.4f},{ci_hi:+.4f}]  "
            f"median Δ={median_diff:+.4f}  "
            f"Wilcoxon p={w_p:.2e} {sig}  "
            f"t-test p={t_p:.2e}  "
            f"Cliff's δ={cd:+.3f}")


def main():
    print("Loading metrics CSVs...")
    unet     = load("unet")
    v2_base  = load("sandwich_v2_lora")
    v2_best  = load("sandwich_v2_lora_drop_a1")
    v1_lora  = load("sandwich_v1_lora")

    pairs = [
        ("V2_best",     v2_best,  "UNet",       unet),
        ("V2_baseline", v2_base,  "UNet",       unet),
        ("V2_best",     v2_best,  "V2_baseline", v2_base),
        ("V2_best",     v2_best,  "V1+LoRA",    v1_lora),
        ("V1+LoRA",     v1_lora,  "UNet",       unet),
    ]

    datasets = ["BUID", "BUSC", "BUSUC", "UDIAT"]

    for name_a, df_a, name_b, df_b in pairs:
        print(f"\n{'='*100}")
        print(f"  Paired test: {name_a} vs {name_b}  (H0: equal Dice)")
        print('='*100)
        # Overall
        print(compare(name_a, df_a, name_b, df_b, dataset_filter=None))
        # Per dataset
        for ds in datasets:
            print(compare(name_a, df_a, name_b, df_b, dataset_filter=ds))

    print("\nLegend: *** p<0.001, ** p<0.01, * p<0.05, . p<0.10, ns not significant")
    print("Effect sizes — Cliff's δ: |0.147| trivial, 0.147–0.33 small, 0.33–0.474 medium, ≥0.474 large")


if __name__ == "__main__":
    main()
