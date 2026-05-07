"""
Render the qualitative comparison figure for the thesis Results chapter.

Selects one external test sample per dataset (BUID/BUSC/BUSUC/UDIAT) where
Sandwich V2 strictly beats all four baselines (U-Net, SAM 3 ZS, SAM LoRA,
Sandwich V1) by Dice, then composes a 4-row x 7-column white-background panel:

    Image | GT | U-Net | SAM 3 ZS | SAM LoRA | Sandwich V1 | Sandwich V2

Output: Thesis-Writing/Figures/qualitative_comparison.png

Usage:
    conda run -n sam3 python BUS-Lesion-Seg/scripts/render_qualitative_figure.py [--verbose]
"""

import argparse
import csv
import os
import sys
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(ROOT, "Experiments")
DATA_DIR = "/home/maia-user/yufeng/Data/BUS_Mix_Seg"
DEFAULT_OUT = "/home/maia-user/yufeng/Thesis-Writing/Figures/qualitative_comparison.png"

MODELS = OrderedDict([
    ("U-Net",        os.path.join(EXP_DIR, "unet")),
    ("SAM 3 ZeroShot",     os.path.join(EXP_DIR, "sam3_original")),
    ("SAM LoRA",     os.path.join(EXP_DIR, "sam3_gtonly_gtonly")),
    ("Sandwich V1",  os.path.join(EXP_DIR, "sandwich_v1_lora")),
    ("Sandwich V2",  os.path.join(EXP_DIR, "sandwich_v2_lora_drop_a1")),
])
EXPECTED_OVERALL_DSC = {
    "U-Net":       0.8851,
    "SAM 3 ZeroShot":    0.1749,
    "SAM LoRA":    0.8364,
    "Sandwich V1": 0.8851,
    "Sandwich V2": 0.8976,
}

DATASETS = ["BUID", "BUSC", "BUSUC", "UDIAT"]

PANEL_W = PANEL_H = 320
COL_HEADER_H = 40
ROW_HEADER_W = 90
DICE_STRIP_H = 26
GUTTER = 4

BG = (255, 255, 255)
TEXT = (20, 20, 20)
GT_COLOR = (60, 160, 60)
GT_CONTOUR = (60, 120, 60)
PRED_COLOR = (220, 60, 60)
PRED_CONTOUR = (180, 30, 30)
ALPHA = 0.40


def load_metrics(exp_dir):
    path = os.path.join(exp_dir, "metrics.csv")
    rows = list(csv.DictReader(open(path)))
    return {(r["Dataset"], r["Filename"]): float(r["Dice"]) for r in rows}


def verify_overall_dsc():
    """Hard-fail if any model's Overall mean Dice deviates from the paper's table."""
    for name, exp_dir in MODELS.items():
        rows = list(csv.DictReader(open(os.path.join(exp_dir, "metrics.csv"))))
        mean = sum(float(r["Dice"]) for r in rows) / len(rows)
        expected = EXPECTED_OVERALL_DSC[name]
        assert abs(mean - expected) < 1e-3, (
            f"{name} Overall DSC {mean:.4f} mismatches paper's {expected:.4f} "
            f"(tolerance 1e-3). Wrong directory? Path={exp_dir}"
        )


def select_samples(verbose=False):
    """Return [(dataset, filename, {model: dice}, gt_area_frac), ...] of length 4."""
    metrics = {name: load_metrics(d) for name, d in MODELS.items()}
    common_keys = set.intersection(*(set(m.keys()) for m in metrics.values()))

    chosen = []
    for ds in DATASETS:
        candidates = []
        for key in common_keys:
            if key[0] != ds:
                continue
            dices = {m: metrics[m][key] for m in MODELS}
            v2 = dices["Sandwich V2"]
            others = [dices[m] for m in MODELS if m != "Sandwich V2"]
            if not all(v2 > o for o in others):
                continue
            label_path = os.path.join(DATA_DIR, "labels", key[1])
            if not os.path.exists(label_path):
                continue
            gt = np.array(Image.open(label_path).convert("L"))
            if gt.max() == 0:
                continue
            gt_area = float((gt > 127).sum()) / gt.size
            candidates.append((key[1], dices, v2 - max(others), gt_area))

        if not candidates:
            print(f"[WARN] No V2-beats-all sample for {ds}; falling back to V2 > 3 baselines",
                  file=sys.stderr)
            for key in common_keys:
                if key[0] != ds:
                    continue
                dices = {m: metrics[m][key] for m in MODELS}
                v2 = dices["Sandwich V2"]
                others = [dices[m] for m in MODELS if m != "Sandwich V2"]
                wins = sum(1 for o in others if v2 > o)
                if wins < 3:
                    continue
                label_path = os.path.join(DATA_DIR, "labels", key[1])
                if not os.path.exists(label_path):
                    continue
                gt = np.array(Image.open(label_path).convert("L"))
                if gt.max() == 0:
                    continue
                gt_area = float((gt > 127).sum()) / gt.size
                candidates.append((key[1], dices, v2 - max(others), gt_area))

        # Rank by V2-vs-second-best gap descending so the figure visually shows the win.
        # Tiebreak by absolute V2 Dice so chosen samples are also strong overall.
        candidates.sort(key=lambda x: (-x[2], -x[1]["Sandwich V2"]))
        fname, dices, gap, gt_area = candidates[0]
        chosen.append((ds, fname, dices, gt_area, gap))

        if verbose:
            print(f"[{ds}] {fname}  GT_area={gt_area*100:.2f}%  V2-2nd_gap={gap:+.4f}")
            for m in MODELS:
                print(f"    {m:<12s} Dice={dices[m]:.4f}")

    return chosen


def overlay_mask(img_arr, mask_arr, color, contour_color):
    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr] * 3, axis=-1)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    out = img_arr.astype(float).copy()
    m = mask_arr > 127
    for c, col in enumerate(color):
        out[m, c] = out[m, c] * (1 - ALPHA) + col * ALPHA
    import scipy.ndimage as ndi
    dilated = ndi.binary_dilation(m, iterations=2)
    edge = dilated & ~m
    out[edge] = list(contour_color)
    return out.astype(np.uint8)


def resize_to(arr, w, h):
    img = Image.fromarray(arr) if isinstance(arr, np.ndarray) else arr
    img = img.convert("RGB")
    img = img.resize((w, h), Image.LANCZOS)
    return np.array(img)


def try_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_centered(draw, text, box_x, box_y, box_w, box_h, font, fill=TEXT):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = box_x + (box_w - tw) // 2 - bbox[0]
    y = box_y + (box_h - th) // 2 - bbox[1]
    draw.text((x, y), text, fill=fill, font=font)


def render_figure(samples, out_path):
    n_rows = len(samples)
    n_cols = 2 + len(MODELS)  # Image + GT + 5 models
    col_names = ["Image", "GT"] + list(MODELS.keys())

    cell_h = PANEL_H + DICE_STRIP_H
    canvas_w = ROW_HEADER_W + n_cols * PANEL_W + (n_cols - 1) * GUTTER
    canvas_h = COL_HEADER_H + n_rows * cell_h + (n_rows - 1) * GUTTER

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)
    font_col = try_font(20)
    font_row = try_font(20)
    font_dice = try_font(20)

    # Column headers
    for j, name in enumerate(col_names):
        x0 = ROW_HEADER_W + j * (PANEL_W + GUTTER)
        draw_centered(draw, name, x0, 0, PANEL_W, COL_HEADER_H, font_col)

    for i, (ds, fname, dices, _, _) in enumerate(samples):
        y_top = COL_HEADER_H + i * (cell_h + GUTTER)

        # Row header (dataset name)
        draw_centered(draw, ds, 0, y_top, ROW_HEADER_W, PANEL_H, font_row)

        img_path = os.path.join(DATA_DIR, "images", fname)
        gt_path = os.path.join(DATA_DIR, "labels", fname)
        img_arr = resize_to(np.array(Image.open(img_path).convert("RGB")), PANEL_W, PANEL_H)
        gt_arr = resize_to(np.array(Image.open(gt_path).convert("L")), PANEL_W, PANEL_H)

        gt_vis = overlay_mask(img_arr, gt_arr, GT_COLOR, GT_CONTOUR)

        cells = [(img_arr, ""), (gt_vis, "GT")]
        for model_name, exp_dir in MODELS.items():
            mask_path = os.path.join(exp_dir, ds, "masks", fname)
            if os.path.exists(mask_path):
                mask_arr = resize_to(np.array(Image.open(mask_path).convert("L")), PANEL_W, PANEL_H)
                vis = overlay_mask(img_arr, mask_arr, PRED_COLOR, PRED_CONTOUR)
            else:
                vis = np.full_like(img_arr, 200)
            cells.append((vis, f"Dice = {dices[model_name]:.3f}"))

        for j, (panel, dice_label) in enumerate(cells):
            x0 = ROW_HEADER_W + j * (PANEL_W + GUTTER)
            canvas.paste(Image.fromarray(panel), (x0, y_top))
            if dice_label:
                draw_centered(draw, dice_label, x0, y_top + PANEL_H,
                              PANEL_W, DICE_STRIP_H, font_dice)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path, dpi=(300, 300), optimize=True)
    print(f"Saved: {out_path}  ({canvas_w}x{canvas_h} px)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    verify_overall_dsc()
    samples = select_samples(verbose=args.verbose)
    render_figure(samples, args.out)


if __name__ == "__main__":
    main()
