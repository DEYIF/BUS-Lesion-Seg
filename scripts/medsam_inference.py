'''
MedSAM (官方通用 medsam_vit_b) 在外部测试集上的零样本推理 / 评测脚本
================================================================

实验设定（已与 yufeng 确认）：
- 模型：bowang-lab/MedSAM 官方通用权重 medsam_vit_b（无超声专用版）。
- 提示：GT 紧致框（Oracle 协议）—— 用 GT mask 的最小外接框作为 box 提示，
        衡量"给定完美定位框后的分割上限"。MedSAM 必须有框提示才能分割。
- 外部测试集：Data/BUS_Mix_Seg 中前缀为 BUSC / BUID / UDIAT / BUSUC 的样本。

预处理 / 推理逻辑严格对齐官方 MedSAM_Inference.py：
  resize 到 1024 (bicubic, anti-aliasing) → astype(uint8) → min-max 归一化到 [0,1]
  → image_encoder → prompt_encoder(boxes) → mask_decoder → sigmoid → 插值回原图 → >0.5。
注意：MedSAM 用 min-max 归一化，而非 SAM 默认的 ImageNet mean/std，切勿混用。

评测工具（get_box_from_mask / Dice / IoU / 扫描逻辑）改编自 src_distill/inference.py，
汇总统计复用 seg_utils/calculate_stats.py。

用法示例：
  # 冒烟测试（只跑 BUSC 前 20 张）
  python scripts/medsam_inference.py --datasets BUSC --limit 20
  # 全量四个外部集
  python scripts/medsam_inference.py
'''
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from skimage import transform
from tqdm import tqdm
from torchvision.ops import masks_to_boxes

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 路径设置 ---
PROJ_ROOT = Path(__file__).resolve().parent.parent          # BUS-Lesion-Seg/
MEDSAM_REPO = "/home/maia-user/yufeng/medsam_repo"          # 官方 segment_anything 所在
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, MEDSAM_REPO)

from segment_anything import sam_model_registry            # noqa: E402
from seg_utils.calculate_stats import calculate_stats_from_csv  # noqa: E402

DEFAULT_DATASETS = ["BUSC", "BUID", "UDIAT", "BUSUC"]


# --- 工具函数（改编自 src_distill/inference.py） ---
def scan_inference_files(root_dir):
    """扫描 images/，到 labels/ 找同名标签；按文件名前缀解析数据集名。"""
    image_dir = Path(root_dir) / "images"
    label_dir = Path(root_dir) / "labels"
    image_files = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    data_list = []
    for img_path in image_files:
        possible_label = label_dir / img_path.name
        label_path = str(possible_label) if possible_label.exists() else None
        data_list.append({
            "image": str(img_path),
            "label": label_path,
            "dataset_name": img_path.name.split('_')[0],
        })
    return data_list


def boxes_from_binary_mask(gt_bin, per_component=False, empty_full_box=False):
    """从二值 mask(H,W) 生成 box 列表 [[x1,y1,x2,y2], ...]（原图坐标）。

    per_component=False：整张前景取单一外接框。
    per_component=True ：按连通域逐个取框（多病灶）。
    空 mask 时：empty_full_box=True 返回全图框 [0,0,W,H]（unet1 模式，框仍喂给 MedSAM）；
               否则返回 None（GT/Oracle 模式，应跳过该图）。
    """
    if gt_bin.sum() == 0:
        if empty_full_box:
            H, W = gt_bin.shape
            return np.array([[0, 0, W, H]], dtype=np.float32)
        return None

    if not per_component:
        m = torch.from_numpy(gt_bin.astype(np.uint8)).unsqueeze(0)  # (1,H,W)
        box = masks_to_boxes(m)[0].numpy()                          # [x1,y1,x2,y2]
        return box[None, :]                                         # (1,4)

    # 连通域分解（用 scipy.ndimage.label）
    from scipy import ndimage
    lab, n = ndimage.label(gt_bin)
    boxes = []
    for i in range(1, n + 1):
        comp = (lab == i)
        if comp.sum() < 10:        # 跳过极小噪点
            continue
        m = torch.from_numpy(comp.astype(np.uint8)).unsqueeze(0)
        boxes.append(masks_to_boxes(m)[0].numpy())
    if not boxes:
        return None
    return np.stack(boxes)         # (N,4)


def compute_dice(pred, gt, smooth=1e-5):
    inter = (pred * gt).sum()
    return float((2. * inter + smooth) / (pred.sum() + gt.sum() + smooth))


def compute_iou(pred, gt, smooth=1e-5):
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return float((inter + smooth) / (union + smooth))


# --- Teacher U-Net（sandwich V2 Stage-1）提示框生成 ---
# 预处理照搬 src_sandwich/precompute_masks_v2.py，忠实复现训练期分布。
def force_rgb_cfirst(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 2:
        img = img[np.newaxis, ...]
    elif img.ndim == 3 and img.shape[-1] <= 4:
        img = np.transpose(img, (2, 0, 1))
    C, H, W = img.shape
    if C == 1:
        img = np.repeat(img, 3, axis=0)
    elif C > 3:
        img = img[:3]
    return img


def build_unet1_predictor(ckpt_path, device):
    """加载 sandwich V2 的 Stage-1 Teacher U-Net (resnet34)，返回 (model, transform)。"""
    from monai.transforms import (Compose, LoadImaged, ScaleIntensityd,
                                  Resized, EnsureTyped, Lambdad)
    from src_unet.model import build_unet
    net = build_unet()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    net.load_state_dict(sd)
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad = False
    tf = Compose([
        LoadImaged(keys=["image"], image_only=False, reverse_indexing=False),
        Lambdad(keys=["image"], func=force_rgb_cfirst),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(512, 512), mode="bilinear"),
        EnsureTyped(keys=["image"]),
    ])
    return net, tf


@torch.no_grad()
def unet1_box(net, tf, img_path, H, W, per_component=False):
    """Teacher U-Net 预测 → 上采样回原图 → 二值 → 框（原图坐标）。
    返回 (boxes(N,4), unet_mask(H,W))；空预测回退全图框。"""
    b = tf({"image": str(img_path)})
    img = b["image"]
    img = img.as_tensor() if hasattr(img, "as_tensor") else torch.as_tensor(img)
    device = next(net.parameters()).device
    img = img.unsqueeze(0).float().to(device)                      # (1,3,512,512)
    prob = torch.sigmoid(net(img))                                 # (1,1,512,512)
    prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
    mask = (prob[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
    boxes = boxes_from_binary_mask(mask, per_component=per_component, empty_full_box=True)
    return boxes, mask


# --- MedSAM 推理（对齐官方 MedSAM_Inference.py） ---
@torch.no_grad()
def medsam_infer_single_box(model, img_embed, box_1024, H, W):
    """单个 box（1024 空间）→ 二值预测 (H,W)。"""
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if box_torch.ndim == 2:
        box_torch = box_torch[:, None, :]   # (B,1,4)
    sparse, dense = model.prompt_encoder(points=None, boxes=box_torch, masks=None)
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(low_res_pred, size=(H, W),
                                 mode="bilinear", align_corners=False)
    return (low_res_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)


@torch.no_grad()
def run_one(model, device, img_path, gt_bin, H, W, boxes_orig):
    """对一张图、给定原图坐标 box 列表，做 MedSAM 推理，返回二值预测 (H,W)。"""
    img_3c = np.array(Image.open(img_path).convert("RGB"))
    # —— 官方预处理 ——
    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None)
    img_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    img_embed = model.image_encoder(img_tensor)             # (1,256,64,64)

    # box 缩放到 1024 空间
    scale = np.array([W, H, W, H], dtype=np.float32)
    pred = np.zeros((H, W), dtype=np.uint8)
    for box in boxes_orig:
        box_1024 = (box / scale * 1024)[None, :]            # (1,4)
        seg = medsam_infer_single_box(model, img_embed, box_1024, H, W)
        pred = np.maximum(pred, seg)                        # 多框取并集
    return pred


def save_overlay(save_path, img_path, gt_bin, pred_bin, boxes_orig, dice):
    img_3c = np.array(Image.open(img_path).convert("RGB"))
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for a in ax:
        a.imshow(img_3c)
        a.axis("off")
    # GT 轮廓（绿）
    ax[0].contour(gt_bin, colors="lime", linewidths=1.2)
    ax[0].set_title("Image + GT (green) + Box (blue)")
    # 预测（红）+ GT 轮廓 + box
    red = np.zeros((*pred_bin.shape, 4))
    red[pred_bin > 0] = [1, 0, 0, 0.45]
    ax[1].imshow(red)
    ax[1].contour(gt_bin, colors="lime", linewidths=1.0)
    ax[1].set_title(f"MedSAM Pred (red)  Dice={dice:.3f}")
    for box in boxes_orig:
        x0, y0, x1, y1 = box
        for a in ax:
            a.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                      edgecolor="blue", facecolor="none", lw=1.5))
    plt.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="MedSAM (medsam_vit_b) 外部测试集推理")
    parser.add_argument("--data_root", type=str,
                        default="/home/maia-user/yufeng/Data/BUS_Mix_Seg")
    parser.add_argument("--ckpt", type=str,
                        default=str(PROJ_ROOT / "checkpoints" / "medsam_vit_b.pth"))
    parser.add_argument("--output_root", type=str,
                        default=str(PROJ_ROOT / "inference_results_medsam"))
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        help="要评测的数据集前缀，默认四个外部集")
    parser.add_argument("--box_source", type=str, default="gt", choices=["gt", "unet1"],
                        help="提示框来源：gt=GT 紧致框(Oracle)；unet1=sandwich V2 Teacher U-Net 预测框")
    parser.add_argument("--unet1_ckpt", type=str,
                        default=str(PROJ_ROOT / "checkpoints" /
                                    "best_teacher_unet_BUSC_BUID_UDIAT_BUSUC.pth"),
                        help="box_source=unet1 时的 Teacher U-Net 权重")
    parser.add_argument("--limit", type=int, default=0,
                        help=">0 时每个数据集最多跑前 N 张（冒烟测试用）")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--per_component", action="store_true",
                        help="按连通域逐病灶给框（默认整张前景单框）")
    parser.add_argument("--no_vis", action="store_true", help="不保存叠加可视化")
    parser.add_argument("--no_mask", action="store_true", help="不保存预测 mask")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_root)
    masks_dir = out_dir / "masks"
    overlays_dir = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_mask:
        masks_dir.mkdir(exist_ok=True)
    if not args.no_vis:
        overlays_dir.mkdir(exist_ok=True)

    print(f"[设备] {device}")
    print(f"[权重] {args.ckpt}")
    print(f"[提示框来源] {args.box_source}")
    model = sam_model_registry["vit_b"](checkpoint=args.ckpt).to(device).eval()

    unet1_net, unet1_tf = None, None
    if args.box_source == "unet1":
        print(f"[Teacher U-Net] {args.unet1_ckpt}")
        unet1_net, unet1_tf = build_unet1_predictor(args.unet1_ckpt, device)

    data_list = scan_inference_files(args.data_root)
    data_list = [d for d in data_list if d["dataset_name"] in args.datasets]
    if args.limit > 0:
        kept, cnt = [], {}
        for d in data_list:
            ds = d["dataset_name"]
            cnt[ds] = cnt.get(ds, 0) + 1
            if cnt[ds] <= args.limit:
                kept.append(d)
        data_list = kept
    print(f"[数据] 待评测 {len(data_list)} 张，数据集={sorted(set(d['dataset_name'] for d in data_list))}")

    rows = []
    skipped = 0
    empty_unet = 0
    for item in tqdm(data_list, desc="MedSAM 推理"):
        if item["label"] is None:
            skipped += 1
            continue
        name = Path(item["image"]).name
        gt = np.array(Image.open(item["label"]))
        if gt.ndim == 3:
            gt = gt[..., 0]
        gt_bin = (gt > 0).astype(np.uint8)
        H, W = gt_bin.shape

        if args.box_source == "gt":
            boxes = boxes_from_binary_mask(gt_bin, per_component=args.per_component)
            if boxes is None:      # 空 GT，Oracle 无法生成框
                skipped += 1
                continue
        else:                      # unet1：用 Teacher U-Net 预测框
            boxes, unet_mask = unet1_box(unet1_net, unet1_tf, item["image"],
                                         H, W, per_component=args.per_component)
            if unet_mask.sum() == 0:
                empty_unet += 1    # U-Net 空预测，已回退全图框

        pred = run_one(model, device, item["image"], gt_bin, H, W, boxes)
        dice = compute_dice(pred, gt_bin)
        iou = compute_iou(pred, gt_bin)
        rows.append({"Filename": name, "Dataset": item["dataset_name"],
                     "Dice": dice, "IoU": iou})

        if not args.no_mask:
            Image.fromarray((pred * 255).astype(np.uint8)).save(masks_dir / name)
        if not args.no_vis:
            save_overlay(overlays_dir / name, item["image"], gt_bin, pred, boxes, dice)

    if not rows:
        print("没有可评测的样本（全部被跳过？）")
        return

    df = pd.DataFrame(rows)
    metrics_path = out_dir / "metrics.csv"
    df.to_csv(metrics_path, index=False)
    print(f"\n[跳过] {skipped} 张（无标签或空 GT）")
    if args.box_source == "unet1":
        print(f"[U-Net 空预测] {empty_unet} 张（已回退全图框喂给 MedSAM）")
    print(f"[保存] 明细 -> {metrics_path}")

    # 控制台汇总
    report = df.groupby("Dataset")[["Dice", "IoU"]].agg(["mean", "std", "count"])
    print("\n===== 各数据集汇总 =====")
    print(report.round(4))
    print("\n===== 总体 =====")
    print(df[["Dice", "IoU"]].agg(["mean", "std", "count"]).round(4))

    # 复用 seg_utils 生成 summary CSV
    summary_path = out_dir / "metrics_summary.csv"
    calculate_stats_from_csv(str(metrics_path), str(summary_path))
    print(f"[保存] 汇总 -> {summary_path}")


if __name__ == "__main__":
    main()
