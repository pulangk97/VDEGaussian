#!/bin/bash
################################################################################
# kitti_dc_train.sh
# 在 Waymo 初步微调的 DynamiCrafter checkpoint 基础上，
# 继续在 KITTI 数据集（0001/0002/0006）上微调，对标 dynamic_train.sh
#
# 关键差异（相对 dynamic_train.sh）：
#   1. data_dir 指向 KITTI 数据目录（而非 Waymo）
#   2. ckpt_path 指向 Waymo 微调后的最新 checkpoint（finetune_from）
#   3. NAME 加 _kitti 后缀，与 Waymo 版本区分
#   4. MAX_STEPS 可适当减少（KITTI 数据量少于 32 个 Waymo 场景）
#
# 用法：
#   cd /path/to/PVG
#   bash scripts/kitti_dc_train.sh
################################################################################

set -e
current_dir=$(pwd)

NUM_GPUS=8
MASTER_PORT=12356        # 与 dynamic_train.sh 的 12355 区分，避免端口冲突
MAX_STEPS=600            # KITTI 3个场景，数据量约为 Waymo 32场景的 1/10，步数相应缩减
NAME="training_512_v1.0"

ORIGINAL_CONFIG="${current_dir}/configs/config_interp_adapt.yaml"
SAVE_ROOT="${current_dir}/checkpoints/kitti_dc"
LOG_DIR="${current_dir}/logs/kitti_dc"

mkdir -p "${SAVE_ROOT}/${NAME}_interp" "${LOG_DIR}"

# ── 检查 KITTI 数据目录 ───────────────────────────────────────────────────────
# DynamiCrafter 的数据加载器读取的是图像，source 应为包含三个场景的父目录
# 需要在 config 中的 data_dir 指向 KITTI image_02 的父目录
# 各场景路径：data/kitti_mot/training/image_02/0001|0002|0006
KITTI_DATA_DIR="${current_dir}/data/kitti_mot/training/image_02"
if [ ! -d "${KITTI_DATA_DIR}" ]; then
    echo "Error: KITTI 数据目录不存在: ${KITTI_DATA_DIR}"
    exit 1
fi
echo "KITTI 数据目录: ${KITTI_DATA_DIR}"

# ── 检查基础配置文件 ──────────────────────────────────────────────────────────
if [ ! -f "${ORIGINAL_CONFIG}" ]; then
    echo "Error: Config 不存在: ${ORIGINAL_CONFIG}"
    exit 1
fi

# ── 找 Waymo 微调后的最新 checkpoint 作为起始点 ──────────────────────────────
WAYMO_DC_CKPT_DIR="${current_dir}/checkpoints/waymo/training_512_v1.0_interp/checkpoints"
if [ ! -d "${WAYMO_DC_CKPT_DIR}" ]; then
    echo "Error: Waymo DC checkpoint 目录不存在: ${WAYMO_DC_CKPT_DIR}"
    echo "请先运行 dynamic_train.sh 完成 Waymo 初步微调。"
    exit 1
fi

# 按 step 编号选取最新 checkpoint（与 waymo_train.sh 的选取逻辑完全一致）
WAYMO_LATEST_CKPT=$(find "${WAYMO_DC_CKPT_DIR}" -name "*.ckpt" -type f | \
    awk -F'step=' '{if(NF>1) print $2+0, $0; else print 0, $0}' | \
    sort -n | tail -1 | awk '{print $2}')

if [ -z "${WAYMO_LATEST_CKPT}" ]; then
    echo "Error: 在 ${WAYMO_DC_CKPT_DIR} 中未找到任何 .ckpt 文件"
    exit 1
fi
echo "基础 checkpoint（Waymo 微调）: ${WAYMO_LATEST_CKPT}"

# ── 生成临时 config ───────────────────────────────────────────────────────────
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT INT TERM
TEMP_CONFIG="${TEMP_DIR}/config_kitti_dc.yaml"
cp "${ORIGINAL_CONFIG}" "${TEMP_CONFIG}"

# 替换 data_dir 为 KITTI 数据目录
sed -i "s|data_dir: \"[^\"]*\"|data_dir: \"${KITTI_DATA_DIR}\"|g" "${TEMP_CONFIG}"

# 替换 max_steps
sed -i "s|max_steps:.*|max_steps: ${MAX_STEPS}  # set by kitti_dc_train.sh|g" "${TEMP_CONFIG}"

# 验证替换结果
echo "config data_dir  : $(grep 'data_dir'  "${TEMP_CONFIG}")"
echo "config max_steps : $(grep 'max_steps' "${TEMP_CONFIG}")"

echo "=========================================="
echo " KITTI DynamiCrafter 微调"
echo " 基于       : Waymo 微调 checkpoint"
echo " 起始 ckpt  : ${WAYMO_LATEST_CKPT}"
echo " 数据目录   : ${KITTI_DATA_DIR}"
echo " GPU 数量   : ${NUM_GPUS}"
echo " MAX_STEPS  : ${MAX_STEPS}"
echo " 保存目录   : ${SAVE_ROOT}"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ── 启动训练 ─────────────────────────────────────────────────────────────────
# 与 dynamic_train.sh 完全一致的启动方式，额外传入 finetune_from 参数
# 指定从 Waymo 微调后的 checkpoint 继续训练，而非从头或从原始预训练权重开始
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
    model.params.ckpt_path="${WAYMO_LATEST_CKPT}" \
    2>&1 | tee "${LOG_DIR}/kitti_dc_train.log"

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[$(date)] KITTI DynamiCrafter 微调完成。checkpoint: ${SAVE_ROOT}"
    echo ""
    echo "后续步骤："
    echo "  1. 检查 checkpoint: ls ${SAVE_ROOT}/${NAME}_interp/checkpoints/"
    echo "  2. 运行 KITTI PVG 训练: bash scripts/kitti_train.sh"
    echo "     （kitti_train.sh 会自动优先使用 KITTI 微调后的 DC checkpoint）"
else
    echo "[$(date)] 训练失败 (exit: ${EXIT_CODE})。日志: ${LOG_DIR}/kitti_dc_train.log"
    exit ${EXIT_CODE}
fi