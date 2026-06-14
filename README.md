# BUS-Lesion-Seg

本项目面向乳腺超声图像中的病灶分割任务，主要目标是探索通用视觉基础模型在医学小样本场景中的迁移能力，并结合 U-Net、SAM3/MedSAM、知识蒸馏和多阶段融合策略，构建更适合 breast ultrasound segmentation 的自动分割流程。

代码按功能模块组织，既包含单独的 U-Net 基线，也包含 SAM3 蒸馏实验、MedSAM 对照实验，以及项目重点方法 Sandwich Architecture。

## 1. 项目结构

```text
BUS-Lesion-Seg/
├── src_unet/              # Teacher U-Net 基线训练、推理、视频/批量推理
├── src_distill/           # SAM3 蒸馏训练与推理
├── src_sandwich/          # Sandwich 多阶段融合模型
├── sam3/                  # SAM3 相关模型代码与工具
├── scripts/               # MedSAM、可视化、消融图、显著性检验等实验脚本
├── seg_utils/             # 指标统计工具
├── notebooks/             # 实验验证 notebook
├── Experiments/           # 各实验输出结果
├── inference_results_*/   # 若干推理结果目录
├── Test_Output/           # 测试输出示例
└── checkpoints/           # 建议放置模型权重，当前仓库未必包含该目录
```

## 2. 环境依赖

建议使用 Python 3.10+ 和 CUDA 环境运行。主要依赖包括：

```bash
pip install torch torchvision
pip install segmentation-models-pytorch monai numpy pandas matplotlib pillow tqdm scikit-learn scikit-image scipy
```

如果运行 SAM3、MedSAM 或 LoRA 相关实验，还需要根据对应模型仓库补齐其依赖和权重文件。代码中部分默认路径指向服务器环境，例如 `/home/maia-user/yufeng/...`，在本地运行前需要替换为自己的数据和权重路径。

## 3. 数据格式约定

大多数脚本默认读取如下数据结构：

```text
DATA_ROOT/
├── images/
│   ├── BUSC_001.png
│   ├── BUID_002.png
│   └── ...
└── labels/
    ├── BUSC_001.png
    ├── BUID_002.png
    └── ...
```

说明：

- `images/` 存放输入图像，常用格式为 `.png` 或 `.jpg`。
- `labels/` 存放同名二值 mask。推理脚本通常允许没有 label，此时只输出预测 mask 和可视化，不计算 Dice/IoU。
- 文件名前缀会被当作数据集名称，例如 `BUSC_001.png` 的数据集名为 `BUSC`。
- `Config.EXTERNAL_TEST_DATASETS` 或命令行参数中的 `target_datasets` 可用于指定外部测试集，例如 `BUSC,BUID,UDIAT,BUSUC`。

Sandwich V2 训练还需要预计算概率图：

```text
DATA_ROOT/
├── sam3_probs_lora/       # SAM3 或 SAM3+LoRA 的 soft probability map
└── unet1_probs/           # Teacher U-Net 的 soft probability map
```

这些目录可通过 `src_sandwich/precompute_masks_v2.py` 生成。

## 4. 模块说明

### 4.1 `src_unet`: Teacher U-Net 基线

该模块训练一个 ResNet34 encoder 的 U-Net，用作基础分割模型，也作为后续 Sandwich 和蒸馏流程中的 Teacher。

主要文件：

- `config.py`: 数据路径、输出路径、训练参数和设备配置。
- `dataset.py`: 数据扫描、划分、MONAI transform、外部测试集划分。
- `model.py`: U-Net 构建函数。
- `train.py`: Teacher U-Net 训练入口。
- `inference.py`: 单模型推理入口，保存 mask、overlay 和指标。
- `inference_video.py`、`inference_video_batch.py`、`inference_us3m.py`: 视频或特定数据源推理脚本。

训练：

```bash
python -m src_unet.train
```

推理：

```bash
python -m src_unet.inference \
  --data-root /path/to/DATA_ROOT \
  --checkpoint checkpoints/best_teacher_unet_BUSC_BUID_UDIAT_BUSUC.pth \
  --output-dir Experiments/unet
```

输出通常包含：

```text
OUTPUT_DIR/
├── metrics.csv
└── BUSC/
    ├── masks/
    └── overlays/
```

### 4.2 `src_distill`: SAM3 蒸馏与 GT-only 微调

该模块使用 Teacher U-Net 为 SAM3 提供 box prompt，并支持两类训练模式：

- `distilled`: 使用 Teacher 输出辅助训练 student SAM3。
- `gtonly`: 只用 GT 监督，作为不使用蒸馏的对照。

主要文件：

- `config.py`: Teacher U-Net、SAM3、数据路径等配置。
- `model.py`: Teacher U-Net 与 SAM3 student 的封装。
- `loss.py`: 分割损失和蒸馏损失。
- `train.py`: SAM3 LoRA/full fine-tuning 训练入口。
- `inference.py`: distilled、gtonly、original 三种模式推理入口。

训练蒸馏版 LoRA：

```bash
python -m src_distill.train \
  --epochs 50 \
  --batch_size 8 \
  --use_distillation \
  --finetune_type lora \
  --exp_tag sam3_distilled
```

训练 GT-only 对照：

```bash
python -m src_distill.train \
  --epochs 50 \
  --batch_size 8 \
  --no_distillation \
  --finetune_type lora \
  --exp_tag sam3_gtonly
```

推理：

```bash
python -m src_distill.inference \
  --mode distilled \
  --finetune_type lora \
  --teacher_ckpt checkpoints/best_teacher_unet_BUSC_BUID_UDIAT_BUSUC.pth \
  --base_sam_ckpt checkpoints/sam3.pt \
  --student_ckpt checkpoints/student_sam3_distilled_lora.pth \
  --data_root /path/to/DATA_ROOT \
  --output_root Experiments/sam3_distilled
```

`--mode original` 可用于评估未微调的 SAM3 基座模型。

### 4.3 `src_sandwich`: Sandwich 多阶段融合模型

Sandwich Architecture 是本项目的核心流程。它将模型分为三个阶段：

1. Stage 1: Teacher U-Net 生成粗定位 mask 或 box prompt。
2. Stage 2: SAM3 根据 prompt 生成更贴合边界的 mask。
3. Stage 3: Student U-Net 融合原图、SAM 输出和 U-Net1 输出，生成最终分割。

V1 使用 `RGB + SAM mask` 的 4 通道输入；V2 改为 `RGB + soft_SAM + soft_UNet1` 的 5 通道输入，更贴近推理时的真实分布。当前推荐优先使用 V2。

#### 4.3.1 预计算 soft mask

训练 V2 前先缓存 SAM3 和 U-Net1 的概率图：

```bash
python -m src_sandwich.precompute_masks_v2 \
  --teacher_ckpt checkpoints/best_teacher_unet_BUSC_BUID_UDIAT_BUSUC.pth \
  --base_sam_ckpt checkpoints/sam3.pt \
  --sam_lora_ckpt checkpoints/student_sam3_gtonly_lora.pth \
  --data_root /path/to/DATA_ROOT \
  --sam_probs_subdir sam3_probs_lora \
  --batch_size 8
```

如果不使用 LoRA SAM3，可设置：

```bash
--sam_lora_ckpt none --sam_probs_subdir sam3_probs
```

#### 4.3.2 训练 Sandwich V2

```bash
python -m src_sandwich.train_v2 \
  --data_root /path/to/DATA_ROOT \
  --sam_prob_subdir sam3_probs_lora \
  --epochs 150 \
  --batch_size 16 \
  --save_path checkpoints/sandwich_unet2_v2.pth \
  --external_test_datasets BUSC,BUID,UDIAT,BUSUC
```

可选参数：

- `--resume`: 从 `save_path` 严格恢复权重、优化器和 epoch。
- `--fine_tune_from`: 只加载已有权重，重新开始新的训练计划。
- `--p_u1_dropout`: 训练时随机置零 U-Net1 概率通道，减少 Student 对 U-Net1 的捷径依赖。

#### 4.3.3 推理 Sandwich V2

```bash
python -m src_sandwich.inference_v2 \
  --teacher_ckpt checkpoints/best_teacher_unet_BUSC_BUID_UDIAT_BUSUC.pth \
  --base_sam_ckpt checkpoints/sam3.pt \
  --unet2_ckpt checkpoints/sandwich_unet2_v2.pth \
  --sam_lora_ckpt checkpoints/student_sam3_gtonly_lora.pth \
  --data_root /path/to/DATA_ROOT \
  --target_datasets BUSC,BUID,UDIAT,BUSUC \
  --output_root Experiments/sandwich_v2_lora
```

自适应回退策略：

- `legacy`: Student 预测为空时回退到 U-Net1；开启 `--enable_a1` 后，当 V2 面积明显大于 V1+LoRA mask 时可回退到 V1 结果。
- `swap`: Student 为空时回退到 SAM3+LoRA；过分割时回退到 U-Net1。

示例：

```bash
python -m src_sandwich.inference_v2 \
  --unet2_ckpt checkpoints/sandwich_unet2_v2.pth \
  --output_root Experiments/sandwich_v2_lora_a1 \
  --enable_a1 \
  --v1_mask_dir Experiments/sandwich_v1_lora
```

### 4.4 `scripts`: 实验、对照与可视化工具

常用脚本：

- `medsam_inference.py`: MedSAM 对照实验，支持 GT box 或 U-Net1 box prompt。
- `calculate_stats.py`: 已放在 `seg_utils/` 中，用于从 `metrics.csv` 生成统计表。
- `visualize_bad_cases.py`: 按 Dice 阈值筛选并可视化 bad cases。
- `significance_test.py`: 对不同方法结果做显著性检验。
- `plot_v2_ablation_bars.py`、`plot_channel_dropout_batch.py`、`plot_hard_vs_soft.py`: 绘制消融实验图。
- `render_qualitative_figure.py`: 生成论文/报告用 qualitative figure。

MedSAM 示例：

```bash
python scripts/medsam_inference.py \
  --data_root /path/to/DATA_ROOT \
  --ckpt checkpoints/medsam_vit_b.pth \
  --output_root inference_results_medsam \
  --datasets BUSC BUID UDIAT BUSUC \
  --box_source gt
```

使用 Teacher U-Net 生成提示框：

```bash
python scripts/medsam_inference.py \
  --box_source unet1 \
  --unet1_ckpt checkpoints/best_teacher_unet_BUSC_BUID_UDIAT_BUSUC.pth
```

## 5. 指标与结果统计

推理脚本通常会在输出目录保存：

```text
OUTPUT_DIR/
├── metrics.csv            # 每张图的 Dice、IoU
├── statistics.csv         # 分数据集和总体统计，部分脚本自动生成
├── metrics_summary.csv    # MedSAM 等脚本的汇总统计
└── <Dataset>/
    ├── masks/             # 二值预测 mask
    └── overlays/          # 原图、预测和 GT 叠加可视化
```

手动统计：

```bash
python -m seg_utils.calculate_stats \
  Experiments/sandwich_v2/metrics.csv \
  Experiments/sandwich_v2/statistics.csv
```

统计字段包括样本数、均值、中位数、方差、标准差、最小值、最大值，以及 Dice/IoU 大于 0.7 的数量和比例。

## 6. 推荐运行流程

从零开始复现实验时，可按以下顺序运行：

1. 准备 `DATA_ROOT/images` 和 `DATA_ROOT/labels`。
2. 修改 `src_unet/config.py`、`src_distill/config.py` 中的默认路径，或在命令行中显式传入路径。
3. 训练 Teacher U-Net：

```bash
python -m src_unet.train
```

4. 可选：训练 SAM3 LoRA 或 GT-only 对照：

```bash
python -m src_distill.train --use_distillation --finetune_type lora --epochs 50
python -m src_distill.train --no_distillation --finetune_type lora --epochs 50
```

5. 预计算 Sandwich V2 所需 soft mask：

```bash
python -m src_sandwich.precompute_masks_v2 \
  --data_root /path/to/DATA_ROOT \
  --teacher_ckpt checkpoints/best_teacher_unet_BUSC_BUID_UDIAT_BUSUC.pth \
  --base_sam_ckpt checkpoints/sam3.pt
```

6. 训练 Sandwich V2：

```bash
python -m src_sandwich.train_v2 \
  --data_root /path/to/DATA_ROOT \
  --save_path checkpoints/sandwich_unet2_v2.pth
```

7. 推理并生成指标：

```bash
python -m src_sandwich.inference_v2 \
  --data_root /path/to/DATA_ROOT \
  --unet2_ckpt checkpoints/sandwich_unet2_v2.pth \
  --output_root Experiments/sandwich_v2
```

## 7. 常见注意事项

- 很多脚本默认使用 `cuda`，没有 GPU 时需要将设备改为 `cpu` 或传入对应参数。
- 本项目中的路径默认值偏向原作者服务器环境，本地运行时请优先检查 `Config.DATA_ROOT`、`CHECKPOINT_DIR`、`SAM_CKPT`、`UNET_CKPT` 等字段。
- `labels/` 文件名应尽量与 `images/` 完全一致，否则训练或指标计算可能无法正确匹配。
- 推理时保存的 mask 通常会被 resize 回原图尺寸；指标多在统一的 512x512 预处理尺寸上计算，便于不同方法公平比较。
- `src_sandwich/train_v2.py` 依赖 `sam3_probs_lora/` 或 `sam3_probs/` 以及 `unet1_probs/`，缺少这些目录时会提示先运行预计算脚本。
- `src_unet/README.md` 当前存在编码显示问题，建议以本 README 为主要使用说明。

## 8. 一句话总结

本仓库以 U-Net 作为医学域定位基线，以 SAM3/MedSAM 作为基础模型对照，并通过 Sandwich Architecture 将 U-Net 的病灶语义定位能力与 SAM 的边界细化能力结合，用于乳腺超声病灶分割的训练、推理、评估和论文实验分析。
