# src_unet/dataset.py
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from monai.data import DataLoader, CacheDataset, Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    Resized, RandFlipd, RandRotate90d, 
    EnsureTyped, Lambdad, Rotate90d
)
from .config import Config
from pathlib import Path
import torch

def get_file_lists(root_dir, use_sam_mask=False, sam_mask_subdir="sam3_masks"):
    """读取 image 和 label 的路径, 可选读取预计算的 sam_mask

    sam_mask_subdir 让 V1 流水线可以切换到 LoRA-SAM 的二值 mask
    （'sam3_masks_lora'）而无需改其他逻辑。
    """
    image_paths = sorted(glob.glob(os.path.join(root_dir, "images", "*.png")))
    label_paths = sorted(glob.glob(os.path.join(root_dir, "labels", "*.png")))

    print(f"找到 {len(image_paths)} 张图像 和 {len(label_paths)} 个标签")
    if len(image_paths) == 0:
        raise ValueError(f"在 {root_dir} 中未找到图像")
    if len(image_paths) != len(label_paths):
        print (len(image_paths), len(label_paths))
        raise ValueError("Image 和 Label 数量不匹配")

    sam_mask_dir = os.path.join(root_dir, sam_mask_subdir)
    if use_sam_mask and not os.path.exists(sam_mask_dir):
        print(f"⚠️ 警告: 请求了使用 sam_mask，但未找到 {sam_mask_dir}，模型可能会报错！")

    data_list = []
    dataset_stats = {} # 用于统计每个数据集有多少张图

    for img_path, lbl_path in zip(image_paths, label_paths):
        # 解析文件名: BrEaST_case010.png -> BrEaST
        filename = os.path.basename(img_path)
        dataset_name = filename.split('_')[0] 
        
        item = {
            "image": img_path, 
            "label": lbl_path, 
            "dataset_name": dataset_name 
        }
        
        if use_sam_mask:
            # Assumes sam mask is saved with identical filename
            item["sam_mask"] = os.path.join(sam_mask_dir, filename)
            
        data_list.append(item)
        
        # 统计
        dataset_stats[dataset_name] = dataset_stats.get(dataset_name, 0) + 1
        
    print(f"📊 数据集分布统计: {dataset_stats}")
    return data_list

def split_by_datasets(all_data, test_dataset_names=Config.EXTERNAL_TEST_DATASETS, val_ratio=0.2, seed=Config.SEED):
    """
    高级划分策略：
    1. 指定某些数据集作为【外部测试集】(External Test)。
    2. 剩下的数据集混合在一起，随机划分为【训练集】和【验证集】(Internal Train/Val)。
    """
    if test_dataset_names is None:
        test_dataset_names = []
        
    train_val_pool = []
    test_pool = []
    
    for item in all_data:
        if item["dataset_name"] in test_dataset_names:
            test_pool.append(item)
        else:
            train_val_pool.append(item)
            
    # 对 train_val_pool 进行随机划分
    from sklearn.model_selection import train_test_split
    train_files, val_files = train_test_split(
        train_val_pool, 
        test_size=val_ratio, 
        random_state=seed
    )
    
    print(f"📌 划分结果:")
    print(f"   - 训练集 (来自 {set(d['dataset_name'] for d in train_files)}): {len(train_files)} 张")
    print(f"   - 验证集 (来自 {set(d['dataset_name'] for d in val_files)}): {len(val_files)} 张")
    if test_pool:
        print(f"   - 外部测试集 (来自 {test_dataset_names}): {len(test_pool)} 张")
        
    return train_files, val_files, test_pool

def force_rgb_cfirst(img):
    """
    将任意维度的图片 (H, W) 或 (H, W, C) 或 (C, H, W)
    强制转换为 (3, H, W) 格式，适配模型输入。
    """
    # 确保是 numpy 数组
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        
    # 情况 1: 纯 2D 灰度图 (H, W) -> (1, H, W)
    if img.ndim == 2:
        img = img[np.newaxis, ...]
        
    # 情况 2: (H, W, C) -> (C, H, W)
    # 判据：最后一个维度小于等于 4 (比如 1, 3, 4)，通常是 Channel
    elif img.ndim == 3 and img.shape[-1] <= 4:
        img = np.transpose(img, (2, 0, 1))
        
    # 此时 img 必定是 (C, H, W)
    C, H, W = img.shape
    
    # 修正通道数: 如果是 1 通道 (灰度)，复制为 3 通道 (RGB)
    if C == 1:
        img = np.repeat(img, 3, axis=0)
    elif C > 3:
        # 如果是 RGBA (4通道)，只取前 3 个
        img = img[:3, ...]
        
    return img

def force_label_1ch(lbl):
    """
    智能处理 Label：
    1. 兼容 UCLM 这种 RGB Mask (红=恶性, 绿=良性)。
    2. 将多通道融合为单通道二值 Mask。
    3. 统一输出格式为 (1, H, W)。
    """
    if isinstance(lbl, torch.Tensor):
        lbl = lbl.detach().cpu().numpy()
        
    # === 情况 A: 多通道图像 (H, W, C) 或 (C, H, W) ===
    if lbl.ndim == 3:
        # 判断通道在哪
        # 如果形状是 (H, W, 3) 或 (H, W, 4) -> 常见情况
        if lbl.shape[-1] in [3, 4]:
            # 核心逻辑：取所有通道的最大值
            # 这样红(255,0,0) -> 255
            # 绿(0,255,0) -> 255
            # 黑(0,0,0) -> 0
            lbl = np.max(lbl, axis=-1)
            
        # 如果形状是 (3, H, W) 或 (4, H, W) -> 少见
        elif lbl.shape[0] in [3, 4]:
            lbl = np.max(lbl, axis=0)

    # === 情况 B: 此时 lbl 已经是 2D (H, W) ===
    # 把它变回 (1, H, W)
    if lbl.ndim == 2:
        lbl = lbl[np.newaxis, ...]
        
    # === 最后一步：二值化 ===
    # 只要像素值 > 0 就视为前景 (1.0)
    # 这能保证红色和绿色都被视为“病灶”
    lbl = (lbl > 0).astype(np.float32)
        
    return lbl

def get_transforms(mode="train", use_sam_mask=False):
    """
    构建 MONAI Transforms
    mode: 'train' (带增强) 或 'val' (确定性)
    """
    keys = ["image", "label"]
    if use_sam_mask:
        keys.append("sam_mask")

    base_transforms = [
        LoadImaged(keys=keys, image_only=False, reverse_indexing=False),
        Lambdad(keys=["image"], func=force_rgb_cfirst),
        Lambdad(keys=["label"], func=force_label_1ch),  # 强制 Label 为单通道二值图
    ]
    
    if use_sam_mask:
        base_transforms.append(Lambdad(keys=["sam_mask"], func=force_label_1ch))
        
    base_transforms.extend([
        # 此时 Label (和 sam_mask) 已经是单通道灰度图，且值为 0/255
        ScaleIntensityd(keys=["image"]),       # image 0-255 -> 0-1
        ScaleIntensityd(keys=["label"]),       # label 0/255 -> 0/1
    ])
    
    if use_sam_mask:
        base_transforms.append(ScaleIntensityd(keys=["sam_mask"]))
        
    base_transforms.append(
        # Resize 统一尺寸
        Resized(
            keys=keys, 
            spatial_size=Config.IMAGE_SIZE, 
            # image用第一根bilinear, 剩下的话如果是label/mask统通用nearest
            mode=["bilinear"] + ["nearest"] * (len(keys)-1)
        )
    )
    
    if mode == "train":
        augmentations = [
            RandFlipd(keys=keys, prob=0.5),
            RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 1)),
        ]
        base_transforms.extend(augmentations)
    
    # 最后转为 Tensor
    base_transforms.append(EnsureTyped(keys=keys))
    
    return Compose(base_transforms)

def force_prob_1ch(arr):
    """
    Convert a probability image to (1, H, W) float WITHOUT binarizing.
    Used for soft SAM / UNet1 probability masks (values should stay in [0,1]).
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 3:
        if arr.shape[-1] <= 4:          # (H, W, C)
            arr = np.transpose(arr, (2, 0, 1))
        arr = arr[0:1]                  # keep only first channel
    return arr.astype(np.float32)


def get_transforms_v2(mode="train"):
    """
    Transforms for Sandwich V2 (5-channel: RGB + soft_SAM + soft_UNet1).
    Loads sam3_probs/ and unet1_probs/ as continuous [0,1] float channels.
    Adds extra augmentation for ultrasound: contrast, Gaussian noise/smooth.
    """
    from monai.transforms import RandAdjustContrastd, RandGaussianNoised, RandGaussianSmoothd
    keys = ["image", "label", "sam_prob", "unet1_prob"]

    transforms = [
        LoadImaged(keys=keys, image_only=False, reverse_indexing=False),
        Lambdad(keys=["image"],      func=force_rgb_cfirst),
        Lambdad(keys=["label"],      func=force_label_1ch),   # binarize GT
        Lambdad(keys=["sam_prob"],   func=force_prob_1ch),    # keep continuous
        Lambdad(keys=["unet1_prob"], func=force_prob_1ch),    # keep continuous
        ScaleIntensityd(keys=["image"]),
        ScaleIntensityd(keys=["label"]),
        ScaleIntensityd(keys=["sam_prob"]),    # 0-255 -> 0-1
        ScaleIntensityd(keys=["unet1_prob"]),  # 0-255 -> 0-1
        Resized(
            keys=keys,
            spatial_size=Config.IMAGE_SIZE,
            mode=["bilinear", "nearest", "bilinear", "bilinear"],
        ),
    ]

    if mode == "train":
        transforms += [
            RandFlipd(keys=keys, prob=0.5),
            RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 1)),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
            RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
            RandGaussianSmoothd(keys=["image"], prob=0.2,
                                sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5)),
        ]

    transforms.append(EnsureTyped(keys=keys))
    return Compose(transforms)


def get_file_lists_v2(root_dir, sam_prob_subdir="sam3_probs", unet1_prob_subdir="unet1_probs"):
    """
    Like get_file_lists() but also includes sam3_probs/ and unet1_probs/ paths.
    Raises a clear error if the precomputed directories are missing.

    sam_prob_subdir/unet1_prob_subdir let the caller point at a specific cache
    (e.g. 'sam3_probs_lora' for LoRA-adapted SAM output) without touching layout.
    """
    image_paths = sorted(glob.glob(os.path.join(root_dir, "images", "*.png")))
    label_paths = sorted(glob.glob(os.path.join(root_dir, "labels", "*.png")))

    sam_prob_dir   = os.path.join(root_dir, sam_prob_subdir)
    unet1_prob_dir = os.path.join(root_dir, unet1_prob_subdir)

    for d in [sam_prob_dir, unet1_prob_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(
                f"Directory not found: {d}\n"
                "Run src_sandwich/precompute_masks_v2.py first."
            )

    print(f"Found {len(image_paths)} images and {len(label_paths)} labels")

    data_list = []
    for img_path, lbl_path in zip(image_paths, label_paths):
        filename     = os.path.basename(img_path)
        dataset_name = filename.split('_')[0]
        data_list.append({
            "image":      img_path,
            "label":      lbl_path,
            "sam_prob":   os.path.join(sam_prob_dir,   filename),
            "unet1_prob": os.path.join(unet1_prob_dir, filename),
            "dataset_name": dataset_name,
        })

    return data_list


def get_dataloaders_v2(data_root=None, sam_prob_subdir="sam3_probs",
                       unet1_prob_subdir="unet1_probs",
                       external_test_datasets=None):
    """
    DataLoaders for Sandwich V2 training (5-channel soft-mask input).
    Expects <sam_prob_subdir>/ and <unet1_prob_subdir>/ under data_root.

    If arguments are None, falls back to Config defaults (backwards-compatible).
    """
    if data_root is None:
        data_root = Config.DATA_ROOT
    if external_test_datasets is None:
        external_test_datasets = Config.EXTERNAL_TEST_DATASETS

    files = get_file_lists_v2(data_root,
                              sam_prob_subdir=sam_prob_subdir,
                              unet1_prob_subdir=unet1_prob_subdir)
    print("V2 配对完成的文件数:", len(files))

    train_files, val_files, _ = split_by_datasets(
        files,
        test_dataset_names=external_test_datasets,
        val_ratio=0.2,
        seed=Config.SEED
    )

    print(f"训练集: {len(train_files)}, 验证集: {len(val_files)}")

    train_ds = Dataset(data=train_files, transform=get_transforms_v2("train"))
    val_ds   = Dataset(data=val_files,   transform=get_transforms_v2("val"))

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE,
                              shuffle=True,  num_workers=Config.NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE,
                              shuffle=False, num_workers=Config.NUM_WORKERS)
    return train_loader, val_loader


def get_dataloaders(use_sam_mask=False, sam_mask_subdir="sam3_masks"):
    files = get_file_lists(Config.DATA_ROOT, use_sam_mask=use_sam_mask,
                           sam_mask_subdir=sam_mask_subdir)
    print("配对完成的文件数:", len(files))
    
    # 划分训练/验证集 (按数据集划分)
    train_files, val_files, test_pool = split_by_datasets(
        files, 
        test_dataset_names=Config.EXTERNAL_TEST_DATASETS,
        val_ratio=0.2,
        seed=Config.SEED
    )
    
    print(f"训练集样本数: {len(train_files)}, 验证集样本数: {len(val_files)}")
    # 创建 Dataset (CacheDataset 可加速训练，如果内存不够改为 Dataset)
    train_ds = Dataset(data=train_files, transform=get_transforms("train", use_sam_mask=use_sam_mask))
    val_ds = Dataset(data=val_files, transform=get_transforms("val", use_sam_mask=use_sam_mask))
    print("数据集创建完成")
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS
    )
    print("数据加载器创建完成")
    return train_loader, val_loader