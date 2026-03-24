#!/bin/bash
################################################################################
# train_unified_dc.sh
# 所有场景数据合并，8 张卡 DDP 训练一个统一的 DynamiCrafter 模型
################################################################################

current_dir=$(pwd)
NUM_GPUS=8
MASTER_PORT=12356
MAX_STEPS=2000
NAME="training_512_v1.0"

ORIGINAL_CONFIG="${current_dir}/configs/config_interp_adapt.yaml"
SAVE_ROOT="${current_dir}/checkpoints/unified_dc"
mkdir -p "${SAVE_ROOT}/${NAME}_interp"

LOG_DIR="${current_dir}/logs/unified_dc"
mkdir -p "${LOG_DIR}"

if [ ! -f "${ORIGINAL_CONFIG}" ]; then
    echo "Error: Config not found at ${ORIGINAL_CONFIG}"
    exit 1
fi

# 检查 data 目录：优先用 ./data/waymo_scenes（多场景父目录）
# 若不存在则报错
DATA_DIR="${current_dir}/data/waymo_scenes"
if [ ! -d "${DATA_DIR}" ]; then
    echo "Error: Data directory not found at ${DATA_DIR}"
    exit 1
fi
echo "Using data directory: ${DATA_DIR}"

# 创建临时 config，将 data_dir 替换为所有场景父目录（与 scene_adaptation.sh 同样的 sed 方式）
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT INT TERM
TEMP_CONFIG="${TEMP_DIR}/config_interp_adapt.yaml"
cp "${ORIGINAL_CONFIG}" "${TEMP_CONFIG}"

# 替换 data_dir（兼容原来的 ./data/waymo_scenes/16 或任何单场景路径）
sed -i "s|data_dir: \"[^\"]*\"|data_dir: \"${DATA_DIR}\"|g" "${TEMP_CONFIG}"

# 替换 max_steps（原始 config 里的值可能是单场景的 500，需覆盖为合并训练的步数）
sed -i "s|max_steps:.*|max_steps: ${MAX_STEPS}  # set by train_unified_dc.sh|g" "${TEMP_CONFIG}"

# 验证替换结果
echo "data_dir in config:  $(grep 'data_dir' ${TEMP_CONFIG})"
echo "max_steps in config: $(grep 'max_steps' ${TEMP_CONFIG})"

echo "=========================================="
echo "Unified DynamiCrafter Training"
echo "GPUs:      ${NUM_GPUS}"
echo "Config:    ${TEMP_CONFIG}"
echo "Data:      ${DATA_DIR}"
echo "Save root: ${SAVE_ROOT}"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --master_addr=127.0.0.1 \
    --master_port=${MASTER_PORT} \
    --node_rank=0 \
    ./submodules/DynamiCrafter/main/trainer.py \
    --base "${TEMP_CONFIG}" \
    --train \
    --name ${NAME}_interp \
    --logdir "${SAVE_ROOT}" \
    --devices ${NUM_GPUS} \
    lightning.trainer.num_nodes=1 \
    2>&1 | tee "${LOG_DIR}/train_unified.log"

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[$(date)] Training completed. Checkpoint: ${SAVE_ROOT}"
else
    echo "[$(date)] Training FAILED (exit: ${EXIT_CODE}). Log: ${LOG_DIR}/train_unified.log"
fi