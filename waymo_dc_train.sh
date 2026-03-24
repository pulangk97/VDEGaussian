#!/bin/bash

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

DATA_DIR="${current_dir}/data/waymo_scenes"
if [ ! -d "${DATA_DIR}" ]; then
    echo "Error: Data directory not found at ${DATA_DIR}"
    exit 1
fi
echo "Using data directory: ${DATA_DIR}"

TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT INT TERM
TEMP_CONFIG="${TEMP_DIR}/config_interp_adapt.yaml"
cp "${ORIGINAL_CONFIG}" "${TEMP_CONFIG}"

sed -i "s|data_dir: \"[^\"]*\"|data_dir: \"${DATA_DIR}\"|g" "${TEMP_CONFIG}"

sed -i "s|max_steps:.*|max_steps: ${MAX_STEPS}  # set by train_unified_dc.sh|g" "${TEMP_CONFIG}"

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