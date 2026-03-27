#!/bin/zsh
# =============================================================
# wait_and_train_lora.sh
# 等待 GPU 显存降至安全阈值后，自动启动 SAM3 LoRA 训练
#
# 用法:
#   bash scripts/wait_and_train_lora.sh [--epochs 10] [--batch_size 2] [--resume]
#
# 参数会透传给 train.py，未指定时使用下面的默认值。
# =============================================================

set -e

# ---------- 可调参数 ----------
MEM_THRESHOLD=5000          # GPU 显存低于此值（MiB）才启动训练
POLL_INTERVAL=60             # 轮询间隔（秒）
EPOCHS=10
BATCH_SIZE=2
FINETUNE_TYPE=lora
EXTRA_ARGS=""                # 其余透传参数，如 "--resume"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)    EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --resume)    EXTRA_ARGS="$EXTRA_ARGS --resume"; shift ;;
        --distill)   EXTRA_ARGS="$EXTRA_ARGS --use_distillation"; shift ;;
        *) shift ;;
    esac
done

# ---------- 路径 ----------
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$REPO_DIR/lora_train.log"
TRAIN_SCRIPT="$REPO_DIR/src_distill/train.py"

# ---------- 等待 GPU 释放 ----------
echo "⏳ 等待 GPU 显存降至 ${MEM_THRESHOLD}MiB 以下... 日志: $LOG" | tee "$LOG"

while true; do
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d " ")
    MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d " ")
    echo "[$(date)] GPU Memory: ${MEM_USED}MiB / ${MEM_TOTAL}MiB" | tee -a "$LOG"

    if [ "$MEM_USED" -lt "$MEM_THRESHOLD" ]; then
        echo "✅ GPU 已释放（${MEM_USED}MiB），准备启动训练..." | tee -a "$LOG"
        break
    fi
    sleep "$POLL_INTERVAL"
done

# ---------- 启动训练 ----------
source activate sam3
export PYTHONPATH=/home/maia-user/yufeng/sam3:$PYTHONPATH

echo "🚀 启动 SAM LoRA 训练: epochs=${EPOCHS}, batch_size=${BATCH_SIZE} ${EXTRA_ARGS}" | tee -a "$LOG"

python "$TRAIN_SCRIPT" \
    --no_distillation \
    --finetune_type "$FINETUNE_TYPE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    $EXTRA_ARGS \
    2>&1 | tee -a "$LOG"
