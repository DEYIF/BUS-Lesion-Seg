import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import segmentation_models_pytorch as smp
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# MONAI 组件
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, 
    Resized, EnsureTyped, Lambdad
)

# 导入配置 (注意：假设你在根目录运行，使用 src_unet 包名)
try:
    from src_unet.config import Config
except ImportError:
    # 兼容在 src_unet 目录下直接运行的情况
    from config import Config

class UnetInferencer:
    def __init__(self, checkpoint_path, device=None):
        self.device = device if device else Config.DEVICE
        print(f"🚀 正在加载模型到 {self.device} ...")
        
        # 1. 加载模型
        self.model = smp.Unet(
            encoder_name=Config.MODEL_ARCH,
            encoder_weights=None, 
            in_channels=3,
            classes=1,
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # 2. 定义预处理管线
        # 技巧：我们定义两套管线，或者利用 allow_missing_keys (但分开写更清晰)
        
        # 管线 A: 有 Label 的情况 (Eval Mode)
        self.transforms_with_label = Compose([
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image"], channel_dim=-1),
            EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
            ScaleIntensityd(keys=["image", "label"]), 
            Resized(keys=["image", "label"], spatial_size=Config.IMAGE_SIZE, mode=["bilinear", "nearest"]),
            EnsureTyped(keys=["image", "label"])
        ])
        
        # 管线 B: 无 Label 的情况 (Pure Inference Mode)
        # self.transforms_image_only = Compose([
        #     LoadImaged(keys=["image"], image_only=False),
        #     EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        #     ScaleIntensityd(keys=["image"]),
        #     Resized(keys=["image"], spatial_size=Config.IMAGE_SIZE, mode="bilinear"),
        #     EnsureTyped(keys=["image"])
        # ])
        self.transforms_image_only = Compose([
            LoadImaged(keys=["image"], image_only=False),
            
            # 1. 强制转为 (3, H, W)
            Lambdad(keys=["image"], func=self.force_rgb_cfirst),
            
            # 2. 归一化 (0-1)
            ScaleIntensityd(keys=["image"]),
            
            # 3. Resize (会对 C, H, W 格式正确缩放)
            Resized(keys=["image"], spatial_size=Config.IMAGE_SIZE, mode="bilinear"),
            
            EnsureTyped(keys=["image"])
        ])

    def force_rgb_cfirst(self, img):
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

    def compute_dice(self, pred, gt):
        smooth = 1e-5
        intersection = (pred * gt).sum()
        return (2. * intersection + smooth) / (pred.sum() + gt.sum() + smooth)

    def compute_iou(self, pred, gt):
        smooth = 1e-5
        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum() - intersection
        return (intersection + smooth) / (union + smooth)

    def run(self, data_item, threshold=0.5):
        """
        核心处理函数：自动判断是否有 Label
        data_item: {'image': path, 'label': path or None, ...}
        """
        has_label = data_item.get("label") is not None
        
        # 1. 预处理
        if has_label:
            data = self.transforms_with_label(data_item)
            gt_tensor = data["label"].unsqueeze(0).to(self.device)
        else:
            # 如果没有 label，我们构建一个只包含 image 的字典给 transforms
            temp_item = {"image": data_item["image"]}
            data = self.transforms_image_only(temp_item)
            gt_tensor = None

        img_tensor = data["image"].unsqueeze(0).to(self.device)

        # 2. 模型推理
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.sigmoid(logits)
            pred_mask = (probs > threshold).float()

        # 3. 数据转回 Numpy (用于可视化和计算)
        pred_np = pred_mask[0, 0].cpu().numpy()
        img_np = img_tensor[0].cpu().numpy().transpose(1, 2, 0) # CHW -> HWC
        
        result = {
            "image": img_np,
            "pred": pred_np,
            "has_label": has_label,
            "filename": os.path.basename(data_item["image"])
        }

        # 4. 如果有 Label，计算指标
        if has_label:
            gt_np = gt_tensor[0, 0].cpu().numpy()
            dice = self.compute_dice(pred_np, gt_np)
            iou = self.compute_iou(pred_np, gt_np)
            
            result.update({
                "gt": gt_np,
                "dice": dice,
                "iou": iou
            })
            
        return result

    def save_prediction_mask(self, pred_np, save_path):
        """
        保存纯净的二值化 Mask
        pred_np: 0 或 1 的 numpy 数组 (float 或 int)
        """
        # 1. 将 0/1 映射到 0/255
        # 乘 255 后，0还是0(黑)，1变成255(白)
        mask_uint8 = (pred_np * 255).astype(np.uint8)
        
        # 2. 转换为 PIL Image 对象 (模式 'L' 表示灰度)
        img = Image.fromarray(mask_uint8, mode='L')
        
        # 3. 保存
        img.save(save_path)

    def visualize(self, res, save_dir):
        """
        动态可视化：
        - 有 GT: 画 3 张图 (原图, GT, 预测)
        - 无 GT: 画 2 张图 (原图, 预测)
        """
        filename = res['filename']
        has_label = res['has_label']
        
        # 根据是否有标签决定子图数量
        cols = 3 if has_label else 2
        plt.figure(figsize=(4 * cols, 4))

        # 1. 原图
        plt.subplot(1, cols, 1)
        plt.title(f"Image: {filename}")
        plt.imshow(res['image'], cmap="gray")
        plt.axis("off")

        # 2. Ground Truth (如果有)
        if has_label:
            plt.subplot(1, cols, 2)
            plt.title("Ground Truth")
            plt.imshow(res['gt'], cmap="gray")
            plt.axis("off")
            
            # 预测图放在第 3 位置
            plot_idx = 3
            title_str = f"Pred (Dice: {res['dice']:.4f})"
        else:
            # 预测图放在第 2 位置
            plot_idx = 2
            title_str = "Prediction (No Label)"

        # 3. 预测叠加图
        plt.subplot(1, cols, plot_idx)
        plt.title(title_str)
        plt.imshow(res['image'], cmap="gray")
        
        # 红色蒙版
        mask_overlay = np.zeros_like(res['image'])
        mask_overlay[:, :, 0] = res['pred'] 
        plt.imshow(mask_overlay, alpha=0.4)
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"res_{filename}"))
        plt.close()

def scan_inference_files(root_dir):
    """
    智能扫描函数：
    - 扫描 images 目录。
    - 尝试去 labels 目录找对应的 label。
    - 找不到 label 不报错，而是标记为 None。
    """
    image_dir = Path(root_dir) / "images"
    label_dir = Path(root_dir) / "labels"
    
    # 支持 png 和 jpg
    image_files = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    
    data_list = []
    
    for img_path in image_files:
        # 尝试寻找对应的 Label (假设同名，且为 png)
        # 你可以根据实际情况修改后缀匹配逻辑
        possible_label_path = label_dir / img_path.name
        
        # 检查 Label 是否存在
        label_path = str(possible_label_path) if possible_label_path.exists() else None
        
        # 解析数据集名称 (用于分组)
        dataset_name = img_path.name.split('_')[0]
        
        data_list.append({
            "image": str(img_path),
            "label": label_path, # 关键点：这里可能是 None
            "dataset_name": dataset_name
        })
        
    return data_list

def main():
    # === 配置区 ===
    # 你的模型路径
    CHECKPOINT = os.path.join(Config.CHECKPOINT_DIR, "best_teacher_unet_allSeg.pth")
    # 想要推理的数据根目录 (可以是训练时的 Data 目录，也可以是新的未标注数据目录)
    # 只要该目录下有 images 文件夹即可
    INFERENCE_ROOT = Config.DATA_ROOT
    OUTPUT_DIR = Config.OUTPUT_DIR
    TEST_DATASETS = Config.EXTERNAL_TEST_DATASETS  # 只推理这些数据集的样本，设为 None 则推理全部
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 扫描文件
    print("📂 正在扫描文件...")
    all_files = scan_inference_files(INFERENCE_ROOT)
    
    # 筛选特定的数据集
    if TEST_DATASETS:
        target_files = [x for x in all_files if x["dataset_name"] in TEST_DATASETS]
    else:
        target_files = all_files
    
    print(f"🔍 找到 {len(target_files)} 个样本")
    num_with_label = sum(1 for x in target_files if x['label'])
    print(f"   - 含标签 (可评估): {num_with_label}")
    print(f"   - 无标签 (仅推理): {len(target_files) - num_with_label}")

    # 2. 初始化
    if not os.path.exists(CHECKPOINT):
        print(f"❌ 模型文件 {CHECKPOINT} 不存在")
        return
    
    inferencer = UnetInferencer(CHECKPOINT)
    
    # 3. 运行
    results_stats = []
    
    pbar = tqdm(target_files, desc="Inference")
    for item in pbar:
        # 运行推理
        res = inferencer.run(item)
        dataset_name = item['dataset_name']
        filename = res['filename'] # e.g., "case01.png"
        # 保存可视化结果与mask
        overlay_dir = os.path.join(OUTPUT_DIR, dataset_name, "overlays")
        mask_dir = os.path.join(OUTPUT_DIR, dataset_name, "masks")
        os.makedirs(overlay_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        inferencer.visualize(res, overlay_dir)

        mask_save_path = os.path.join(mask_dir, filename)
        inferencer.save_prediction_mask(res['pred'], mask_save_path)
        
        # 只有当计算了指标时，才记录到统计表
        if res['has_label']:
            results_stats.append({
                "Dataset": dataset_name,
                "Filename": filename,
                "Dice": res['dice'],
                "IoU": res['iou']
            })
            pbar.set_postfix({"Dice": f"{res['dice']:.4f}"})
        else:
            pbar.set_postfix({"Status": "Pred"})

    # 4. 生成报告 (仅针对有标签的数据)
    if results_stats:
        df = pd.DataFrame(results_stats)
        print("\n" + "="*40)
        print("🏆 评估报告 (仅包含带标签数据)")
        print("="*40)
        report = df.groupby("Dataset")[["Dice", "IoU"]].agg(['mean', 'std', 'count'])
        print(report)
        df.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)
        print(f"\n📄 指标已保存至 {OUTPUT_DIR}/metrics.csv")
    else:
        print("\n🚀 所有数据均为无标签推理，未生成指标报告。可视化结果已保存。")

if __name__ == "__main__":
    main()