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
import torch.nn.functional as F

# MONAI 组件（仅保留必要部分）
from monai.transforms import Compose

# 导入配置 (注意：假设你在根目录运行，使用 src_unet 包名)
try:
    from src_unet.config import Config
except ImportError:
    # 兼容在 src_unet 目录下直接运行的情况
    from config import Config

class UnetInferencer:
    def __init__(self, checkpoint_path, device=None):
        if device:
            self.device = device
        else:
            cfg_device = Config.DEVICE
            if cfg_device == "cuda" and not torch.cuda.is_available():
                print("⚠️  Config.DEVICE='cuda' 但 CUDA 不可用，已自动切换到 CPU")
                self.device = "cpu"
            else:
                self.device = cfg_device
        print(f"🚀 正在加载模型到 {self.device} ...")

        # 1. 加载模型
        self.model = smp.Unet(
            encoder_name=Config.MODEL_ARCH,
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # 预处理尺寸 (H, W)
        self.img_size = Config.IMAGE_SIZE  # e.g. (512, 512)

    def preprocess_image(self, image_path):
        """用 PIL 加载图像并预处理为 (3, H, W) float tensor，归一化到 [0,1]。"""
        H, W = self.img_size
        img = Image.open(image_path).convert('RGB')        # 统一为 3 通道 RGB
        img = img.resize((W, H), Image.BILINEAR)           # PIL.resize 参数为 (W, H)
        arr = np.array(img, dtype=np.float32) / 255.0      # (H, W, 3)，归一化
        return torch.from_numpy(arr).permute(2, 0, 1)      # (3, H, W)

    def preprocess_label(self, label_path):
        """用 PIL 加载标签并预处理为 (1, H, W) float tensor，值为 0 或 1。"""
        H, W = self.img_size
        lbl = Image.open(label_path).convert('L')          # 灰度
        lbl = lbl.resize((W, H), Image.NEAREST)            # 最近邻，保持二值边界
        arr = (np.array(lbl, dtype=np.float32) > 0).astype(np.float32)  # 0/255 -> 0/1
        return torch.from_numpy(arr).unsqueeze(0)          # (1, H, W)

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
        核心处理函数：推理并还原回原始尺寸
        """
        has_label = data_item.get("label") is not None

        # 获取原始尺寸和 DPI（PIL.Image.size 返回 (W, H)）
        with Image.open(data_item["image"]) as _img:
            orig_w, orig_h = _img.size
            orig_dpi = _img.info.get("dpi", None)  # 用于保存 mask 时保持 DPI 一致

        # 1. 预处理（PIL → tensor，resize 到 512×512）
        img_tensor_512 = self.preprocess_image(data_item["image"]).unsqueeze(0).to(self.device)
        gt_tensor_512 = None
        if has_label:
            gt_tensor_512 = self.preprocess_label(data_item["label"]).unsqueeze(0).to(self.device)

        # 2. 模型推理 (在 512x512 上进行)
        with torch.no_grad():
            logits = self.model(img_tensor_512)
            probs_512 = torch.sigmoid(logits) # (1, 1, 512, 512)

        # === ✨ 关键步骤 2: 还原回原始尺寸 ===
        
        # A. 还原概率图 (使用双线性插值 bilinear，结果更平滑)
        # 注意：必须在 threshold 之前做插值，否则边缘会有锯齿
        probs_orig = F.interpolate(
            probs_512, 
            size=(orig_h, orig_w), 
            mode="bilinear", 
            align_corners=False
        )
        
        # B. 还原输入图像 (用于可视化背景)
        img_tensor_orig = F.interpolate(
            img_tensor_512,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        )
        
        # C. 生成最终 Mask (在原图尺寸上切阈值)
        pred_mask_orig = (probs_orig > threshold).float()

        # 3. 数据转回 Numpy
        # 此时取出来的数据已经是原始尺寸 (H_orig, W_orig)
        pred_np = pred_mask_orig[0, 0].cpu().numpy()
        
        # Image: (C, H, W) -> (H, W, C)
        img_np = img_tensor_orig[0].cpu().numpy().transpose(1, 2, 0)
        
        result = {
            "image": img_np,
            "pred": pred_np,
            "has_label": has_label,
            "filename": os.path.basename(data_item["image"]),
            "dpi": orig_dpi,
        }

        # 4. 如果有 Label，计算指标
        # ⚠️ 注意：计算指标通常建议在“模型训练尺寸(512)”上算，或者把 GT 读进来不 Resize 直接算。
        # 为了简单且公平，这里我们通常在 512 尺寸上算指标（反映模型能力），但在原图尺寸上保存结果（反映应用效果）。
        if has_label:
            # 我们在 512 尺度上计算指标 (因为 gt_tensor 目前是 512 的)
            # 如果你想在原图尺度算，需要重新加载原始 Label，会比较慢
            pred_mask_512 = (probs_512 > threshold).float()
            pred_np_512 = pred_mask_512[0, 0].cpu().numpy()
            gt_np_512 = gt_tensor_512[0, 0].cpu().numpy()
            
            dice = self.compute_dice(pred_np_512, gt_np_512)
            iou = self.compute_iou(pred_np_512, gt_np_512)
            
            # 可视化用的 GT 也需要还原回去以便对齐
            # 这里的 Label 使用 'nearest' 插值，因为它是 0/1 标签
            gt_tensor_orig = F.interpolate(
                gt_tensor_512,
                size=(orig_h, orig_w),
                mode="nearest" 
            )
            gt_np_orig = gt_tensor_orig[0, 0].cpu().numpy()
            
            result.update({
                "gt": gt_np_orig, # 存原始尺寸的 GT 用于可视化
                "dice": dice,     # 存 512 尺度的指标
                "iou": iou
            })
            
        return result

    def save_prediction_mask(self, pred_np, save_path, dpi=None):
        """
        保存纯净的二值化 Mask
        pred_np: 0 或 1 的 numpy 数组 (float 或 int)
        dpi: 原图的 DPI 元数据，保持一致以便在 PPT 等软件中叠放时尺寸对齐
        """
        mask_uint8 = (pred_np * 255).astype(np.uint8)
        img = Image.fromarray(mask_uint8, mode='L')
        save_kwargs = {}
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        img.save(save_path, **save_kwargs)

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
    import argparse
    parser = argparse.ArgumentParser(description="UNet 推理脚本")
    parser.add_argument("--data-root", type=str, default=Config.DATA_ROOT,
                        help="数据根目录，需包含 images/ 子文件夹（默认: Config.DATA_ROOT）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录（默认: <data-root>/masks_inference）")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(Config.CHECKPOINT_DIR, "best_teacher_unet_BUSC_BUID_UDIAT_BUSUC.pth"),
                        help="模型权重路径")
    args = parser.parse_args()

    CHECKPOINT = args.checkpoint
    INFERENCE_ROOT = args.data_root
    OUTPUT_DIR = args.output_dir if args.output_dir else os.path.join(INFERENCE_ROOT, "masks_inference")
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
        inferencer.save_prediction_mask(res['pred'], mask_save_path, dpi=res.get('dpi'))
        
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