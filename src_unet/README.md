# U-Net 推理功能使用说明 (Inference Guide)

本文档介绍了如何配置和运行 `src_unet` 目录下的推理脚本 `inference.py`。该脚本主要用于加载训练好的 U-Net 模型，对超声图像（不论是否带有标签）进行病灶分割，并将预测结果还原为原始图像尺寸保存。

## 1. 环境依赖

确保你已经安装了以下主要的 Python 库：
- `torch`, `torchvision` (PyTorch)
- `segmentation_models_pytorch` (SMP 模型库)
- `monai` (用于医学图像预处理)
- `numpy`, `pandas`, `matplotlib`, `Pillow`, `tqdm`

## 2. 目录结构要求

推理时，输入数据的根目录 (`DATA_ROOT`) 必须包含一个 `images` 文件夹（存放测试图像）。如果有对应的 Ground Truth 标签，可以存放在 `labels` 文件夹中（可选）。

```text
DATA_ROOT/
├── images/
│   ├── BUSC_001.png
│   ├── BUID_002.jpg
│   └── ... 
└── labels/  (可选)
    ├── BUSC_001.png
    ├── BUID_002.png
    └── ...
```
> **注意**：脚本会根据文件名的第一部分（如 `BUSC`_001.png）来提取数据集名称，以用于按数据集保存结果和统计指标。

## 3. 配置参数 (`config.py`)

在运行推理前，请检查并修改 `src_unet/config.py` 中的相关路径：

- `DATA_ROOT`: 待推理数据的根目录路径（参照上述结构）。
- `OUTPUT_DIR`: 推理结果（预测掩膜和可视化叠加图）的保存路径。
- `CHECKPOINT_DIR`: 存放模型权重（`.pth`）的目录。
- `EXTERNAL_TEST_DATASETS`: 需要测试的数据集名称列表（例如 `["BUSC", "BUID"]`）。
  - *如果想处理 `DATA_ROOT` 下的**所有**图像，可以在此设为空列表 `[]` 或 `None`。*

**模型权重路径提醒**：
目前在 `inference.py` 的 `main()` 函数中，模型文件名被默认写死为 `best_teacher_unet_BUSC_BUID_UDIAT_BUSUC.pth`。如果你的模型文件名不同，请直接在 `inference.py` 底部修改 `CHECKPOINT` 变量的拼接逻辑：
```python
# inference.py 的 main() 函数中
CHECKPOINT = os.path.join(Config.CHECKPOINT_DIR, "你的模型名称.pth")
```

## 4. 运行推理

建议在项目根目录（`BUS-Lesion-Seg/`）下运行，以便正确导入包模块：

```bash
# 在 BUS-Lesion-Seg 目录下执行
python -m src_unet.inference
```

或者直接进入 `src_unet` 目录运行：

```bash
cd src_unet
python inference.py
```

## 5. 推理输出说明

运行完毕后，在 `OUTPUT_DIR` 中会按照【数据集名称】生成以下结构：

```text
OUTPUT_DIR/
├── metrics.csv                # (仅存在带有标签的数据时) 各个数据集的平均指标(Dice, IoU)
├── BUSC/                      # 按照每个数据集自动分子目录
│   ├── masks/                 # 保存对应原始图像尺寸的 0/255 二值化预测掩膜图
│   └── overlays/              # 保存可视化图像（原图 + 红色预测区域叠加，若有GT也包含）
└── BUID/
    ├── masks/
    └── overlays/
```

### 特性简介
1. **自动尺寸复原**：无论原图大小尺寸如何，脚本会自动在预处理时 Resize 到指定尺寸 (如 512x512) 进行推理，但最后保存的掩膜 (`masks`) 和可视化图 (`overlays`) 都会通过差值**无损还原回原图自身的长宽像素**。
2. **兼容无标签数据**：如果在 `labels` 文件夹中找不到同名标签（或者干脆没有 `labels` 目录），脚本依然会正常前向推理病输出掩膜和预测可视化，此时跳过 指标(Dice/IoU) 的计算。
