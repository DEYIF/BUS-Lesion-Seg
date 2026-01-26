import os
import glob
from tqdm import tqdm

# === ⚙️ 配置区 ===
# 数据根目录
DATA_ROOT = "Data/BUS_Mix_Cls" 

# 要处理的特定后缀 (根据你的描述是 STU)
TARGET_SUFFIX = "STU" 

# ⚠️ 安全开关：True = 只打印不修改 | False = 真的改名
DRY_RUN = True
# =================

def process_folder(folder_name):
    dir_path = os.path.join(DATA_ROOT, folder_name)
    print(f"\n📂 正在扫描 {folder_name} ...")
    
    # 查找所有包含该后缀的文件
    # 逻辑：文件名以 "_STU.png" 结尾
    pattern = f"*_{TARGET_SUFFIX}.png"
    files = glob.glob(os.path.join(dir_path, pattern))
    
    if not files:
        print(f"   未找到符合 '*{TARGET_SUFFIX}.png' 模式的文件。")
        return

    count = 0
    for file_path in files:
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        
        # 解析文件名: case01_STU.png
        name_no_ext = os.path.splitext(filename)[0] # case01_STU
        
        # 按照下划线分割
        parts = name_no_ext.split('_')
        
        # 确保格式符合预期 (至少有两部分，且最后一部分是 STU)
        if len(parts) >= 2 and parts[-1] == TARGET_SUFFIX:
            # === 核心逻辑：调换位置 ===
            # 取出最后一部分 (STU) 作为前缀
            new_prefix = parts[-1]
            # 取出前面的部分 (case01) 作为后缀
            rest_part = "_".join(parts[:-1])
            
            # 组合新名字: STU_case01.png
            new_filename = f"{new_prefix}_{rest_part}.png"
            new_file_path = os.path.join(directory, new_filename)
            
            if DRY_RUN:
                print(f"   [预览] {filename}  -->  {new_filename}")
            else:
                os.rename(file_path, new_file_path)
                
            count += 1
            
    if DRY_RUN:
        print(f"✅ 扫描结束，共发现 {count} 个文件需要重命名。(未实际执行)")
    else:
        print(f"✅ 执行结束，已重命名 {count} 个文件。")

if __name__ == "__main__":
    print(f"🚀 开始重命名任务 (目标后缀: {TARGET_SUFFIX})")
    if DRY_RUN:
        print("⚠️  当前为 DRY RUN (空跑) 模式，不会修改任何文件！")
        print("⚠️  确认输出无误后，请将代码中的 DRY_RUN = True 改为 False 再运行。")
    
    # 1. 处理 Images
    process_folder("images")
    
    # 2. 处理 Labels (必须同步修改，否则就这就对不上了！)
    # process_folder("labels")