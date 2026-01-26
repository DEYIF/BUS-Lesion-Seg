from monai.transforms import LoadImaged, EnsureChannelFirstd, Compose
import torch
import os
import glob

data_root = "Data/BUS_Mix_Seg"  # 请根据实际路径修改
# 模拟一条数据
sample_data = {
    "image": os.path.join(data_root, "images", os.listdir(os.path.join(data_root, "images"))[0]),
    "label": os.path.join(data_root, "labels", os.listdir(os.path.join(data_root, "labels"))[0])
}

print("\n=== 2. MONAI Pipeline 逐步调试 ===")

# --- 测试 A: 只加载 (LoadImage) ---
loader = LoadImaged(keys=["image", "label"], image_only=True)
data_loaded = loader(sample_data)

img_raw = data_loaded["image"]
lbl_raw = data_loaded["label"]

print(f"Raw Load Shape (Image): {img_raw.shape}")
print(f"Raw Load Shape (Label): {lbl_raw.shape}")

# --- 诊断 A ---
# 如果 Image 是 (H, W, 3)，说明是 RGB 图，通道在最后。
# 如果 Image 是 (H, W)，说明是灰度图，没有通道维度。

# --- 测试 B: 增加通道维度 (EnsureChannelFirst) ---
# ⚠️ 这里是重点！我们要测试 channel_dim 参数的影响
try:
    # 你的代码里可能写的是 channel_dim="no_channel"
    channel_adder = EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel")
    data_channeled = channel_adder(data_loaded)
    
    print(f"After 'no_channel' (Image): {data_channeled['image'].shape}")
    print(f"After 'no_channel' (Label): {data_channeled['label'].shape}")
    
    # 🚨 这里的坑：
    # 如果原图是 (H, W, 3)，用 "no_channel" 会变成 (1, H, W, 3) -> 这是一个 4D Tensor！
    # Resize 变换看到 4D Tensor，会以为是 (C, D, H, W) 或者是 3D 图像，导致报错。
    
except Exception as e:
    print(f"❌ EnsureChannelFirstd 报错: {e}")

# --- 测试 C: 正确的 RGB 处理方式 ---
if len(img_raw.shape) == 3 and img_raw.shape[-1] == 3:
    print("\n✅ 检测到 RGB 输入，尝试正确的 channel_dim=-1 ...")
    # 对于 RGB (H, W, 3)，通道是最后一个维度，所以要填 -1
    correct_adder = EnsureChannelFirstd(keys=["image"], channel_dim=-1)
    # 对于 Label (H, W)，通常没有通道，所以还是 no_channel
    label_adder = EnsureChannelFirstd(keys=["label"], channel_dim="no_channel")
    
    data_corrected = correct_adder(data_loaded)
    data_corrected = label_adder(data_corrected)
    
    print(f"Corrected Shape (Image): {data_corrected['image'].shape} -> 应该是 (3, H, W)")
    print(f"Corrected Shape (Label): {data_corrected['label'].shape} -> 应该是 (1, H, W)")