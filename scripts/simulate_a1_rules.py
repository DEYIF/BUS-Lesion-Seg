"""
Offline A1 rule simulator.

For each test sample, load V1+LoRA mask, V2+LoRA mask, and GT.
Apply various A1 rules to produce an A1-adjusted output, recompute Dice.
Reports per-rule: # fired, Overall Dice, bad cases (Dice<0.7), and how many
of the 19 V1-saves-V2 retreat cases got rescued.

Usage:
    conda run -n sam3 python BUS-Lesion-Seg/scripts/simulate_a1_rules.py
"""
import csv
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

ROOT = Path("/home/maia-user/yufeng/BUS-Lesion-Seg")
EXP  = ROOT / "Experiments"
DATA = Path("/home/maia-user/yufeng/Data/BUS_Mix_Seg")
THR  = 0.7
EPS  = 1e-5


def load_metrics(p):
    return {r["Filename"]: (r["Dataset"], float(r["Dice"]))
            for r in csv.DictReader(open(p))}


def dice(pred, gt):
    inter = (pred * gt).sum()
    return (2.0 * inter + EPS) / (pred.sum() + gt.sum() + EPS)


def iou(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return inter / max(union, 1)


def load_mask(path, target_shape=None):
    arr = np.array(Image.open(path).convert("L"))
    if target_shape and arr.shape != target_shape:
        arr = np.array(
            Image.fromarray(arr).resize((target_shape[1], target_shape[0]), Image.NEAREST)
        )
    return arr > 127  # bool


def candidate_rules():
    """Return dict name → predicate(meta) -> bool to use V1."""
    rules = {}

    # R0: original (under-segmentation only)
    def r0(m):
        return m["v1_area"] > 0.005 * m["img_size"] and \
               m["v2_area"] < 0.3 * m["v1_area"]
    rules["R0_under03"] = r0

    # R1: over-segmentation, k=2
    def r1(m):
        return m["v1_area"] > 0.005 * m["img_size"] and \
               m["v2_area"] > 2.0 * m["v1_area"]
    rules["R1_over2"] = r1

    # R1.5: over-seg, k=1.5
    def r15(m):
        return m["v1_area"] > 0.005 * m["img_size"] and \
               m["v2_area"] > 1.5 * m["v1_area"]
    rules["R1.5_over1.5"] = r15

    # R2: low IoU(V1,V2) + V1 confident
    def r2(m):
        return m["v1_area"] > 0.005 * m["img_size"] and \
               m["iou_v1_v2"] < 0.5
    rules["R2_iou0.5"] = r2

    # R3: low IoU + V1 confident + V2 over (combined)
    def r3(m):
        return m["v1_area"] > 0.005 * m["img_size"] and \
               m["iou_v1_v2"] < 0.5 and \
               m["v2_area"] > 1.5 * m["v1_area"]
    rules["R3_iou0.5+over1.5"] = r3

    # R4: very disagreement either direction (IoU < 0.3)
    def r4(m):
        return m["v1_area"] > 0.005 * m["img_size"] and \
               m["iou_v1_v2"] < 0.3
    rules["R4_iou0.3"] = r4

    # R5: combined under + over disagreement, with IoU
    def r5(m):
        if m["v1_area"] <= 0.005 * m["img_size"]:
            return False
        ratio = m["v2_area"] / max(m["v1_area"], 1)
        return (ratio < 0.3 or ratio > 2.0) and m["iou_v1_v2"] < 0.5
    rules["R5_under0.3orover2+iou"] = r5

    # R6: over k=1.5, but require V1 confident (>1% image)
    def r6(m):
        return m["v1_area"] > 0.01 * m["img_size"] and \
               m["v2_area"] > 1.5 * m["v1_area"]
    rules["R6_over1.5+v1>=1%"] = r6

    # R7: over k=1.5 AND IoU < 0.6
    def r7(m):
        return m["v1_area"] > 0.005 * m["img_size"] and \
               m["v2_area"] > 1.5 * m["v1_area"] and \
               m["iou_v1_v2"] < 0.6
    rules["R7_over1.5+iou0.6"] = r7

    # R8: over ratio adaptive — for tiny V1 (< 5%) require larger over ratio
    def r8(m):
        if m["v1_area"] <= 0.005 * m["img_size"]:
            return False
        v1_pct = m["v1_area"] / m["img_size"]
        ratio = m["v2_area"] / m["v1_area"]
        if v1_pct < 0.05:    # small lesion: stricter
            return ratio > 2.0
        else:                # larger lesion: looser
            return ratio > 1.5
    rules["R8_adaptive"] = r8

    # R9: hybrid — use V1 if V2 mass disagreement, but use mean of V1+V2 as middle ground
    # (kept for reference, not used)

    # R10: high over ratio AND IoU < 0.5
    def r10(m):
        return m["v1_area"] > 0.005 * m["img_size"] and \
               m["v2_area"] > 2.0 * m["v1_area"] and \
               m["iou_v1_v2"] < 0.5
    rules["R10_over2+iou0.5"] = r10

    # R11: per-dataset filter — only enable on BUSUC/BUSC where V2 retreats (avoid BUID/UDIAT)
    # placeholder; needs dataset name in meta

    return rules


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2_dir", default="sandwich_v2_lora",
                        help="Subdir under Experiments/ holding the V2 mask outputs to evaluate.")
    parser.add_argument("--v1_dir", default="sandwich_v1_lora",
                        help="Subdir under Experiments/ holding the V1+LoRA mask outputs (fallback source).")
    parser_args = parser.parse_args()

    print(f"Loading metrics... (V2={parser_args.v2_dir}, V1={parser_args.v1_dir})")
    v1_metrics = load_metrics(EXP / parser_args.v1_dir / "metrics.csv")
    v2_metrics = load_metrics(EXP / parser_args.v2_dir / "metrics.csv")

    target_files = list(v2_metrics.keys())
    print(f"Total samples: {len(target_files)}")

    # Identify the 19 retreat cases
    retreat_set = set()
    for fn, (ds, d) in v2_metrics.items():
        v1d = v1_metrics.get(fn, (None, -1))[1]
        if d < THR and v1d >= THR:
            retreat_set.add(fn)
    print(f"V1+LoRA saves V2+LoRA retreat cases: {len(retreat_set)}")

    rules = candidate_rules()

    # Per-rule stats
    stats = {name: {
        "fired": 0,
        "fired_filenames": [],
        "fired_was_better": 0,
        "fired_was_worse": 0,
        "retreat_rescued": 0,
    } for name in rules}

    new_dice = {name: {} for name in rules}

    skipped = 0
    print(f"Simulating {len(rules)} rules over {len(target_files)} samples...")
    for fn in target_files:
        ds = v2_metrics[fn][0]
        v2_mask_path = EXP / parser_args.v2_dir / ds / "masks" / fn
        v1_mask_path = EXP / parser_args.v1_dir / ds / "masks" / fn
        gt_path      = DATA / "labels" / fn

        if not (v2_mask_path.exists() and v1_mask_path.exists() and gt_path.exists()):
            skipped += 1
            continue

        v2_mask = load_mask(v2_mask_path)
        v1_mask = load_mask(v1_mask_path, target_shape=v2_mask.shape)
        gt_mask = load_mask(gt_path,      target_shape=v2_mask.shape)

        v1_area  = float(v1_mask.sum())
        v2_area  = float(v2_mask.sum())
        img_size = float(v1_mask.size)
        iou_vv   = iou(v1_mask, v2_mask) if (v1_area + v2_area) > 0 else 1.0

        meta = {
            "v1_area":   v1_area,
            "v2_area":   v2_area,
            "img_size":  img_size,
            "iou_v1_v2": iou_vv,
        }

        v2_dice = dice(v2_mask.astype(np.float32), gt_mask.astype(np.float32))
        v1_dice = dice(v1_mask.astype(np.float32), gt_mask.astype(np.float32))

        for name, pred in rules.items():
            try:
                fire = pred(meta)
            except Exception:
                fire = False
            if fire:
                final_dice = v1_dice
                stats[name]["fired"] += 1
                stats[name]["fired_filenames"].append(fn)
                if v1_dice > v2_dice:
                    stats[name]["fired_was_better"] += 1
                elif v1_dice < v2_dice:
                    stats[name]["fired_was_worse"] += 1
                if fn in retreat_set and v1_dice >= THR:
                    stats[name]["retreat_rescued"] += 1
            else:
                final_dice = v2_dice
            new_dice[name][fn] = final_dice

    # Baseline (no rule)
    baseline = {fn: v2_metrics[fn][1] for fn in target_files}
    bd_mean  = np.mean(list(baseline.values()))
    bd_bad   = sum(1 for d in baseline.values() if d < THR)

    print(f"\n{'='*90}")
    print(f"{'Rule':30s} {'Fired':>6s} {'Better':>7s} {'Worse':>6s} {'Same':>5s} "
          f"{'Rescued':>8s} {'Mean DSC':>10s} {'Δmean':>8s} {'Bad':>5s} {'Δbad':>5s}")
    print('='*90)
    print(f"{'V2 baseline (no A1)':30s} {'-':>6s} {'-':>7s} {'-':>6s} {'-':>5s} "
          f"{'-':>8s} {bd_mean:>10.4f} {'-':>8s} {bd_bad:>5d} {'-':>5s}")
    for name in rules:
        s = stats[name]
        new_mean = np.mean(list(new_dice[name].values()))
        new_bad  = sum(1 for d in new_dice[name].values() if d < THR)
        same = s["fired"] - s["fired_was_better"] - s["fired_was_worse"]
        print(f"{name:30s} {s['fired']:>6d} {s['fired_was_better']:>7d} "
              f"{s['fired_was_worse']:>6d} {same:>5d} "
              f"{s['retreat_rescued']:>4d}/{len(retreat_set):>3d} "
              f"{new_mean:>10.4f} {new_mean - bd_mean:>+8.4f} {new_bad:>5d} {new_bad - bd_bad:>+5d}")

    if skipped:
        print(f"\nSkipped (missing files): {skipped}")


if __name__ == "__main__":
    main()
