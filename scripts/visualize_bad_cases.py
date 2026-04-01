"""
Visualize bad cases (Dice < threshold) for UNet and SAM3 models side-by-side.

Output: BUS-Lesion-Seg/Experiments/bad_case_analysis/
  - per_case/<filename>.png  -- 5-panel: Image | GT | UNet | SAM3 | Sandwich
  - summary_unet_bad.png     -- grid of cases where UNet fails
  - summary_sam3_bad.png     -- grid of cases where SAM3 fails
  - bad_cases_report.txt     -- text analysis with failure categories

Usage:
  conda run -n sam3 python scripts/visualize_bad_cases.py [--threshold 0.3]
"""

import csv
import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── paths ──────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR    = os.path.join(ROOT, "Experiments")
DATA_DIR   = "/home/maia-user/yufeng/Data/BUS_Mix_Seg"
OUT_DIR    = os.path.join(EXP_DIR, "bad_case_analysis")

MODELS = {
    "UNet":     os.path.join(EXP_DIR, "unet"),
    "SAM3":     os.path.join(EXP_DIR, "sam3_gtonly_gtonly"),
    "SAM3-KD":  os.path.join(EXP_DIR, "sam3_distilled"),
    "Sandwich": os.path.join(EXP_DIR, "sandwich_original_sam3"),
}
PANEL_NAMES = ["Image", "GT"] + list(MODELS.keys())
PANEL_W, PANEL_H = 300, 300
HEADER_H = 36
TEXT_COLOR = (220, 220, 220)
BG_COLOR   = (30, 30, 30)

# ── helpers ────────────────────────────────────────────────────────────────

def load_metrics(exp_dir):
    path = os.path.join(exp_dir, "metrics.csv")
    rows = list(csv.DictReader(open(path)))
    return {r["Filename"]: (r["Dataset"], float(r["Dice"]), float(r["IoU"])) for r in rows}


def overlay_mask_on_image(img_arr, mask_arr, color=(255, 80, 80), alpha=0.45):
    """Blend a binary mask (0/255) over an RGB image. img_arr must be HxWx3."""
    # Ensure img is RGB (H,W,3)
    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr] * 3, axis=-1)
    # Ensure mask is 2D
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    out = img_arr.copy().astype(float)
    m = mask_arr > 127
    for c, col in enumerate(color):
        out[m, c] = out[m, c] * (1 - alpha) + col * alpha
    # Draw contour (yellow border)
    import scipy.ndimage as ndi
    dilated = ndi.binary_dilation(m, iterations=2)
    edge = dilated & ~m
    out[edge] = [255, 200, 0]
    return out.astype(np.uint8)


def resize_to(arr, w, h):
    img = Image.fromarray(arr) if isinstance(arr, np.ndarray) else arr
    img = img.convert("RGB")
    img = img.resize((w, h), Image.LANCZOS)
    return np.array(img)


def try_font(size=14):
    for name in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if os.path.exists(name):
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def draw_label(canvas, text, x, y, font, color=TEXT_COLOR):
    draw = ImageDraw.Draw(canvas)
    draw.text((x, y), text, fill=color, font=font)


def make_panel_row(filename, dataset, metrics_per_model, threshold):
    """
    Returns a PIL image: [Image | GT | UNet | SAM3 | SAM3-KD | Sandwich]
    """
    img_path   = os.path.join(DATA_DIR, "images", filename)
    label_path = os.path.join(DATA_DIR, "labels", filename)

    if not os.path.exists(img_path) or not os.path.exists(label_path):
        return None

    img_arr   = resize_to(np.array(Image.open(img_path).convert("RGB")), PANEL_W, PANEL_H)
    label_arr = resize_to(np.array(Image.open(label_path).convert("L")),  PANEL_W, PANEL_H)
    gt_vis    = overlay_mask_on_image(img_arr, label_arr, color=(80, 180, 80))

    panels = [img_arr, gt_vis]
    dice_labels = ["", "GT"]

    for model_name, exp_dir in MODELS.items():
        mask_path = os.path.join(exp_dir, dataset, "masks", filename)
        if os.path.exists(mask_path):
            mask_arr = resize_to(np.array(Image.open(mask_path).convert("L")), PANEL_W, PANEL_H)
            vis      = overlay_mask_on_image(img_arr, mask_arr, color=(255, 80, 80))
        else:
            vis = np.full_like(img_arr, 50)  # dark placeholder

        dice = metrics_per_model.get(model_name, {}).get(filename, (None, None, None))
        dice_val = dice[1] if dice[1] is not None else -1
        panels.append(vis)
        dice_labels.append(f"{model_name}\nDice={dice_val:.3f}" if dice_val >= 0 else model_name)

    n = len(panels)
    total_w = n * PANEL_W
    total_h = PANEL_H + HEADER_H * 2
    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    font_big  = try_font(13)
    font_small = try_font(11)

    # Title row
    draw = ImageDraw.Draw(canvas)
    title = f"{filename}  [{dataset}]"
    draw.text((4, 4), title, fill=(255, 220, 80), font=font_big)

    for i, (panel, dlabel) in enumerate(zip(panels, dice_labels)):
        canvas.paste(Image.fromarray(panel), (i * PANEL_W, HEADER_H))
        lines = dlabel.split("\n")
        for j, line in enumerate(lines):
            col = TEXT_COLOR
            if "Dice=" in line:
                val = float(line.split("=")[1])
                col = (255, 100, 100) if val < threshold else (100, 220, 100)
            draw.text((i * PANEL_W + 4, HEADER_H + PANEL_H + 2 + j * 14),
                      line, fill=col, font=font_small)

    return canvas


def make_summary_grid(rows, title, out_path, cols=3):
    if not rows:
        return
    col_w = PANEL_W * len(PANEL_NAMES)
    row_h = PANEL_H + HEADER_H * 2

    # column headers
    header_h = 24
    n_rows = (len(rows) + cols - 1) // cols
    # Actually lay them vertically (one row per case)
    grid_h = len(rows) * row_h + header_h
    grid = Image.new("RGB", (col_w, grid_h), BG_COLOR)
    draw = ImageDraw.Draw(grid)
    font = try_font(14)
    draw.text((4, 4), title, fill=(255, 220, 80), font=font)

    for i, row_img in enumerate(rows):
        grid.paste(row_img, (0, header_h + i * row_h))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.save(out_path)
    print(f"Saved: {out_path}")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Dice threshold for 'bad case' (default: 0.3)")
    args = parser.parse_args()
    threshold = args.threshold

    os.makedirs(os.path.join(OUT_DIR, "per_case"), exist_ok=True)

    # Load all metrics
    metrics_per_model = {}
    for model_name, exp_dir in MODELS.items():
        try:
            metrics_per_model[model_name] = load_metrics(exp_dir)
        except FileNotFoundError:
            print(f"Warning: metrics not found for {model_name}")
            metrics_per_model[model_name] = {}

    unet_metrics = metrics_per_model["UNet"]
    sam3_metrics = metrics_per_model["SAM3"]

    all_filenames = set(unet_metrics.keys())

    # Categorize bad cases
    categories = {
        "both_bad":        [],  # both UNet and SAM3 fail
        "only_unet_bad":   [],  # UNet fails, SAM3 OK
        "only_sam3_bad":   [],  # SAM3 fails, UNet OK
        "empty_gt":        [],  # GT has no lesion (annotation edge case)
    }

    for fname in sorted(all_filenames):
        dataset, unet_dice, _ = unet_metrics[fname]
        sam3_entry = sam3_metrics.get(fname)
        sam3_dice  = sam3_entry[1] if sam3_entry else None

        # Check if GT is empty
        label_path = os.path.join(DATA_DIR, "labels", fname)
        label_arr  = np.array(Image.open(label_path).convert("L")) if os.path.exists(label_path) else None
        gt_empty   = (label_arr is not None and label_arr.max() == 0)

        if gt_empty:
            categories["empty_gt"].append(fname)
        elif unet_dice < threshold and (sam3_dice is None or sam3_dice < threshold):
            categories["both_bad"].append(fname)
        elif unet_dice < threshold and sam3_dice >= threshold:
            categories["only_unet_bad"].append(fname)
        elif unet_dice >= threshold and sam3_dice is not None and sam3_dice < threshold:
            categories["only_sam3_bad"].append(fname)

    # ── per-case panels ────────────────────────────────────────────────────
    bad_files = set()
    for cat_cases in categories.values():
        bad_files.update(cat_cases)

    per_case_panels = {}
    for fname in sorted(bad_files):
        dataset = unet_metrics.get(fname, sam3_metrics.get(fname, (None,)))[0]
        panel = make_panel_row(fname, dataset, metrics_per_model, threshold)
        if panel is not None:
            out_path = os.path.join(OUT_DIR, "per_case", fname)
            panel.save(out_path)
            per_case_panels[fname] = panel

    # ── summary grids ──────────────────────────────────────────────────────
    def collect_panels(fnames):
        return [per_case_panels[f] for f in fnames if f in per_case_panels]

    make_summary_grid(
        collect_panels(categories["both_bad"]),
        f"Both UNet & SAM3 fail (Dice < {threshold})",
        os.path.join(OUT_DIR, "summary_both_bad.png"),
    )
    make_summary_grid(
        collect_panels(categories["only_unet_bad"]),
        f"Only UNet fails (Dice < {threshold}), SAM3 succeeds",
        os.path.join(OUT_DIR, "summary_only_unet_bad.png"),
    )
    make_summary_grid(
        collect_panels(categories["only_sam3_bad"]),
        f"Only SAM3 fails (Dice < {threshold}), UNet succeeds",
        os.path.join(OUT_DIR, "summary_only_sam3_bad.png"),
    )
    make_summary_grid(
        collect_panels(categories["empty_gt"]),
        "Empty GT mask (no lesion annotated)",
        os.path.join(OUT_DIR, "summary_empty_gt.png"),
    )

    # ── text report ───────────────────────────────────────────────────────
    report_path = os.path.join(OUT_DIR, "bad_cases_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Bad Case Analysis  (threshold = Dice < {threshold})\n")
        f.write("=" * 60 + "\n\n")

        totals = {m: 0 for m in MODELS}
        for model_name, mm in metrics_per_model.items():
            bad = [v for v in mm.values() if v[1] < threshold]
            totals[model_name] = len(bad)

        f.write("Bad case counts per model:\n")
        for m, cnt in totals.items():
            f.write(f"  {m:12s}: {cnt:3d} / {len(unet_metrics)}\n")
        f.write("\n")

        for cat_name, fnames in categories.items():
            f.write(f"── {cat_name.upper()} ({len(fnames)} cases) ──\n")
            for fname in sorted(fnames):
                dataset = unet_metrics.get(fname, sam3_metrics.get(fname, (None,)))[0]
                line = f"  {dataset:6s}  {fname:35s}"
                for model_name, mm in metrics_per_model.items():
                    entry = mm.get(fname)
                    d = entry[1] if entry else -1
                    line += f"  {model_name}={d:.3f}"
                # Check prediction size vs GT
                label_path = os.path.join(DATA_DIR, "labels", fname)
                if os.path.exists(label_path):
                    l_arr = np.array(Image.open(label_path).convert("L"))
                    label_nz = np.count_nonzero(l_arr)
                    gt_area_pct = 100 * label_nz / l_arr.size
                    line += f"  GT_area={gt_area_pct:.1f}%"
                    # Guess failure mode
                    unet_mask_path = os.path.join(MODELS["UNet"], dataset, "masks", fname)
                    if os.path.exists(unet_mask_path):
                        u_arr = np.array(Image.open(unet_mask_path).convert("L"))
                        u_nz = np.count_nonzero(u_arr)
                        if u_nz == 0 and label_nz > 0:
                            line += "  [UNet: FALSE NEGATIVE]"
                        elif u_nz > 0 and label_nz == 0:
                            line += "  [UNet: FALSE POSITIVE]"
                f.write(line + "\n")
            f.write("\n")

    print(f"Report saved: {report_path}")
    print(f"\nDone. Output in: {OUT_DIR}")


if __name__ == "__main__":
    main()
