"""
Compose the thesis figure `bad_case_categories.png` from per-case panels.

Selects 2 representative samples per category (only_sam3_bad, only_unet_bad,
both_bad, v2_regression) and stitches them into a single multi-row figure.
Crops out the SAM 3 zero-shot column (uninformative -- 99.6% bad cases) so
each row shows 6 panels: Image | GT | U-Net | SAM_LoRA | Sandwich V1 | Sandwich V2.

Output: Thesis-Writing/Figures/bad_case_categories.png
"""

import os
from PIL import Image, ImageDraw, ImageFont

PER_CASE_DIR = "/home/maia-user/yufeng/BUS-Lesion-Seg/Experiments/bad_case_analysis_v2_dice08/per_case"
OUT_PATH     = "/home/maia-user/yufeng/Thesis-Writing/Figures/bad_case_categories.png"

PANEL_W = 300
ROW_H   = 372  # height of a per_case panel (includes header + dice labels)

# Crop: keep cols 0,1,2 (Image,GT,U-Net) and 4,5,6 (SAM LoRA, V1, V2); drop col 3 (SAM 3)
KEEP_COLS = [0, 1, 2, 4, 5, 6]

CATEGORY_HEADER_H = 38   # height of the category title bar above each pair of rows
FIG_BG = (30, 30, 30)
FIG_FG = (250, 220, 80)

# (category_label, per_case filename) -- one representative sample per category
ROWS = [
    ("(a) only_sam3_bad  —  U-Net succeeds, SAM_LoRA fails  (cascade follows U-Net)",
     "BUID_102_Benign.png"),
    ("(b) only_unet_bad  —  U-Net fails, SAM_LoRA succeeds  (cascade follows SAM_LoRA)",
     "UDIAT_000019.png"),
    ("(c) both_bad  —  both baselines fail  (Sandwich must reconstruct)",
     "BUSUC_00402.png"),
    ("(d) v2_regression  —  both baselines succeed, Sandwich V2 retreats below threshold",
     "BUSUC_0017.png"),
]


def try_font(size):
    for name in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if os.path.exists(name):
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def crop_panel(img):
    """Drop the SAM 3 column (col 3). Returns a 6-column-wide image."""
    out_w = len(KEEP_COLS) * PANEL_W
    out = Image.new("RGB", (out_w, img.height), FIG_BG)
    for new_idx, src_col in enumerate(KEEP_COLS):
        crop = img.crop((src_col * PANEL_W, 0, (src_col + 1) * PANEL_W, img.height))
        out.paste(crop, (new_idx * PANEL_W, 0))
    return out


def main():
    out_w = len(KEEP_COLS) * PANEL_W
    total_h = sum(CATEGORY_HEADER_H + ROW_H for _ in ROWS)
    canvas = Image.new("RGB", (out_w, total_h), FIG_BG)
    draw = ImageDraw.Draw(canvas)
    font = try_font(18)

    y = 0
    for cat_label, fname in ROWS:
        # Category header bar
        draw.rectangle([(0, y), (out_w, y + CATEGORY_HEADER_H)], fill=(50, 50, 50))
        draw.text((10, y + 10), cat_label, fill=FIG_FG, font=font)
        y += CATEGORY_HEADER_H
        src_path = os.path.join(PER_CASE_DIR, fname)
        if not os.path.exists(src_path):
            print(f"WARNING: missing {src_path}")
            y += ROW_H
            continue
        panel = Image.open(src_path)
        cropped = crop_panel(panel)
        canvas.paste(cropped, (0, y))
        y += ROW_H

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    canvas.save(OUT_PATH, optimize=True)
    print(f"Saved: {OUT_PATH}  size={canvas.size}")


if __name__ == "__main__":
    main()
