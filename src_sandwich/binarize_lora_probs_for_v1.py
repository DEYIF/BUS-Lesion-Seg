"""
Convert LoRA-SAM soft probability masks (uint8 0-255) into binary masks (0/255)
for V1 (4-channel student) training, without rerunning SAM inference.

Source: /home/maia-user/yufeng/Data/BUS_Mix_Seg/sam3_probs_lora/<filename>.png  (uint8 [0,255])
Target: /home/maia-user/yufeng/Data/BUS_Mix_Seg/sam3_masks_lora/<filename>.png  (uint8 0/255)
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

SRC_DIR = Path("/home/maia-user/yufeng/Data/BUS_Mix_Seg/sam3_probs_lora")
DST_DIR = Path("/home/maia-user/yufeng/Data/BUS_Mix_Seg/sam3_masks_lora")
THRESHOLD = 127  # uint8 threshold equivalent to soft prob > 0.5 (~127.5/255)


def main():
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SRC_DIR}")
    DST_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(list(SRC_DIR.glob("*.png")))
    print(f"Found {len(files)} soft-prob PNGs in {SRC_DIR}")

    n_skipped = 0
    for src in tqdm(files, desc="binarize"):
        dst = DST_DIR / src.name
        if dst.exists():
            n_skipped += 1
            continue
        arr = np.array(Image.open(src).convert("L"))
        binary = (arr > THRESHOLD).astype(np.uint8) * 255
        Image.fromarray(binary, mode="L").save(dst)

    print(f"Done. Wrote to {DST_DIR}. Skipped (existed) {n_skipped}.")


if __name__ == "__main__":
    main()
