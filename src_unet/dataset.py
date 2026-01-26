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

def get_file_lists(root_dir):
    """读取 image 和 label 的路径"""
    image_paths = sorted(glob.glob(os.path.join(root_dir, "images", "*.png")))
    label_paths = sorted(glob.glob(os.path.join(root_dir, "labels", "*.png")))
    
    print(f"找到 {len(image_paths)} 张图像 和 {len(label_paths)} 个标签")
    if len(image_paths) == 0:
        raise ValueError(f"在 {root_dir} 中未找到图像")
    if len(image_paths) != len(label_paths):
        print (len(image_paths), len(label_paths))
        raise ValueError("Image 和 Label 数量不匹配")

    data_list = []
    dataset_stats = {} # 用于统计每个数据集有多少张图

    for img_path, lbl_path in zip(image_paths, label_paths):
        # 解析文件名: BrEaST_case010.png -> BrEaST
        filename = os.path.basename(img_path)
        dataset_name = filename.split('_')[0] 
        
        # 存入字典
        data_list.append({
            "image": img_path, 
            "label": lbl_path, 
            "dataset_name": dataset_name # <--- 新增的标签
        })
        
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

def get_transforms(mode="train"):
    """
    构建 MONAI Transforms
    mode: 'train' (带增强) 或 'val' (确定性)
    """
    base_transforms = [
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),  # RGB 图通道在最后
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        
        # 此时 Label 已经是单通道灰度图，且值为 0/255
        # 只需要简单的归一化即可
        ScaleIntensityd(keys=["image"]),       # image 0-255 -> 0-1
        ScaleIntensityd(keys=["label"]),       # label 0/255 -> 0/1
        
        # Resize 统一尺寸
        Resized(
            keys=["image", "label"], 
            spatial_size=Config.IMAGE_SIZE, 
            mode=["bilinear", "nearest"] 
        ),

        #Rotate90d(keys=["image", "label"], k=3),

    ]
    
    # if mode == "train":
    #     augmentations = [
    #         RandFlipd(keys=["image", "label"], prob=0.5),
    #         RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
    #     ]
    #     base_transforms.extend(augmentations)
    
    # 最后转为 Tensor
    base_transforms.append(EnsureTyped(keys=["image", "label"]))
    
    return Compose(base_transforms)

def get_dataloaders():
    files = get_file_lists(Config.DATA_ROOT)
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
    train_ds = Dataset(data=train_files, transform=get_transforms("train"))
    val_ds = Dataset(data=val_files, transform=get_transforms("val"))
    print("数据集创建完成")
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS
    )
    print("数据加载器创建完成")
    return train_loader, val_loader