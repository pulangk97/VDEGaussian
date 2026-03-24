#!/bin/bash
################################################################################
# kitti_vde_train.sh
# VDEGaussian KITTI 训练脚本（含 DynamiCrafter 蒸馏）
# GPU 调度逻辑与 pvg_train.sh 完全一致：可自选 GPU，有空闲就执行下一个场景
#
# 数据目录结构（与 PVG README 一致）：
#   data/kitti_mot/training/
#     calib/    0001.txt  0002.txt  0006.txt
#     image_02/ 0001/<frame_id>.png  ...
#     image_03/ 0001/<frame_id>.png  ...
#     sky_02/   0001/<frame_id>.png  ...
#     sky_03/   0001/<frame_id>.png  ...
#     oxts/     0001.txt  0002.txt  0006.txt
#     velodyne/ 0001/<frame_id>.bin  ...
#
# 用法：
#   cd /path/to/PVG
#   bash scripts/kitti_vde_train.sh
################################################################################

# ── 场景列表：每行格式为 "scene_id start_frame end_frame" ─────────────────────
scene_defs=(
    "0001 181 446"
    "0002   0  232"
    "0006  0 269"
)

# ── 可用 GPU 列表（按需修改，例如只用 0 1 2） ─────────────────────────────────
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

current_dir=$(pwd)

# ── 路径配置 ──────────────────────────────────────────────────────────────────
DATA_ROOT="${current_dir}/data/kitti_mot/training"
OUTPUT_ROOT="${current_dir}/eval_output/kitti_vde"
CONFIG="${current_dir}/configs/kitti_nvs.yaml"
DC_CONFIG="${current_dir}/configs/config_interp_adapt.yaml"

# DynamiCrafter checkpoint：优先 KITTI 微调版，回退 Waymo 微调版
KITTI_DC_CKPT_DIR="${current_dir}/checkpoints/kitti_dc/training_512_v1.0_interp/checkpoints"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "${current_dir}/logs/kitti_vde_train"
> "${current_dir}/logs/kitti_vde_completed.txt"
> "${current_dir}/logs/kitti_vde_failed.txt"

# ── 选取最新 DynamiCrafter checkpoint（按 step 编号数值排序）────────────────
pick_dc_ckpt() {
    local ckpt_dir=""
    if [ -d "${KITTI_DC_CKPT_DIR}" ] && \
       [ -n "$(find "${KITTI_DC_CKPT_DIR}" -name '*.ckpt' 2>/dev/null)" ]; then
        ckpt_dir="${KITTI_DC_CKPT_DIR}"
        echo "[INFO] 使用 KITTI 微调 DC checkpoint: ${ckpt_dir}" >&2
    elif [ -d "${WAYMO_DC_CKPT_DIR}" ] && \
         [ -n "$(find "${WAYMO_DC_CKPT_DIR}" -name '*.ckpt' 2>/dev/null)" ]; then
        ckpt_dir="${WAYMO_DC_CKPT_DIR}"
        echo "[INFO] 使用 Waymo 微调 DC checkpoint: ${ckpt_dir}" >&2
    else
        echo "[WARN] 未找到任何 DynamiCrafter checkpoint，将以纯 PVG 模式训练" >&2
        echo ""; return
    fi
    find "${ckpt_dir}" -name "*.ckpt" -type f | \
        awk -F'step=' '{if(NF>1) print $2+0, $0; else print 0, $0}' | \
        sort -n | tail -1 | awk '{print $2}'
}

LATEST_DC_CKPT=$(pick_dc_ckpt)

echo "=========================================="
echo "VDEGaussian Training - KITTI Scenes"
echo "Scenes : ${#scene_defs[@]}"
echo "GPUs   : ${GPUS[*]}"
echo "Config : ${CONFIG}"
echo "Output : ${OUTPUT_ROOT}"
if [ -n "${LATEST_DC_CKPT}" ]; then
    echo "DC ckpt: ${LATEST_DC_CKPT}"
else
    echo "DC ckpt: None（纯 PVG 模式）"
fi
echo "=========================================="

################################################################################
# 函数：训练单个场景
################################################################################
train_single_scene() {
    local scene_id=$1
    local sf=$2
    local ef=$3
    local gpu_id=$4
    local log_file=$5

    local source_path="${DATA_ROOT}/image_02/${scene_id}"
    local model_path="${OUTPUT_ROOT}/${scene_id}_${sf}_${ef}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [GPU-${gpu_id}] START  scene=${scene_id} frames=${sf}~${ef}"

    if [ ! -d "${source_path}" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [GPU-${gpu_id}] ERROR  source_path not found: ${source_path}"
        echo "${scene_id}_${sf}_${ef}" >> "${current_dir}/logs/kitti_vde_failed.txt"
        return 1
    fi

    mkdir -p "${model_path}"

    if [ -n "${LATEST_DC_CKPT}" ]; then
        CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
            --config "${CONFIG}" \
            source_path="${source_path}" \
            model_path="${model_path}" \
            start_frame=${sf} \
            end_frame=${ef} \
            vdm_ckp_dir="${LATEST_DC_CKPT}" \
            vdm_config_dir="${DC_CONFIG}" \
            vdm_weight=1.0 \
            >> "${log_file}" 2>&1
    else
        CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
            --config "${CONFIG}" \
            source_path="${source_path}" \
            model_path="${model_path}" \
            start_frame=${sf} \
            end_frame=${ef} \
            >> "${log_file}" 2>&1
    fi

    local exit_code=$?
    if [ ${exit_code} -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [GPU-${gpu_id}] DONE   scene=${scene_id} frames=${sf}~${ef}"
        echo "${scene_id}_${sf}_${ef}" >> "${current_dir}/logs/kitti_vde_completed.txt"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [GPU-${gpu_id}] FAILED scene=${scene_id} frames=${sf}~${ef} (exit=${exit_code})"
        echo "${scene_id}_${sf}_${ef}" >> "${current_dir}/logs/kitti_vde_failed.txt"
    fi
    return ${exit_code}
}

################################################################################
# 并行调度（与 pvg_train.sh 完全一致：有空闲 GPU 就执行下一个场景）
################################################################################

scene_idx=0
declare -A gpu_pids
declare -A gpu_scenes

# ── 初始化：每个 GPU 分配第一个场景 ──────────────────────────────────────────
for gpu_id in "${GPUS[@]}"; do
    [ ${scene_idx} -ge ${#scene_defs[@]} ] && break

    read -r scene_id sf ef <<< "${scene_defs[$scene_idx]}"
    ((scene_idx++))

    log_file="${current_dir}/logs/kitti_vde_train/${scene_id}_${sf}_${ef}_gpu${gpu_id}.log"
    train_single_scene "${scene_id}" ${sf} ${ef} ${gpu_id} "${log_file}" &
    gpu_pids[$gpu_id]=$!
    gpu_scenes[$gpu_id]="${scene_id}_${sf}_${ef}"

    echo "Assigned scene ${scene_id}[${sf}~${ef}] → GPU-${gpu_id} (PID=${gpu_pids[$gpu_id]}, scene_idx=${scene_idx})"
    sleep 3
done

echo "------------------------------------------"
echo "Initial batch launched. scene_idx=${scene_idx}/${#scene_defs[@]}"
echo "------------------------------------------"

# ── 持续调度：有 GPU 空闲就分配下一个场景 ─────────────────────────────────────
while [ ${scene_idx} -lt ${#scene_defs[@]} ]; do
    sleep 30

    for gpu_id in "${GPUS[@]}"; do
        [ -z "${gpu_pids[$gpu_id]}" ] && continue
        [ ${scene_idx} -ge ${#scene_defs[@]} ] && break

        if ! kill -0 "${gpu_pids[$gpu_id]}" 2>/dev/null; then
            read -r scene_id sf ef <<< "${scene_defs[$scene_idx]}"
            ((scene_idx++))

            log_file="${current_dir}/logs/kitti_vde_train/${scene_id}_${sf}_${ef}_gpu${gpu_id}.log"
            echo "GPU-${gpu_id} free (was: ${gpu_scenes[$gpu_id]}), assigning ${scene_id}[${sf}~${ef}] (scene_idx=${scene_idx})"

            train_single_scene "${scene_id}" ${sf} ${ef} ${gpu_id} "${log_file}" &
            gpu_pids[$gpu_id]=$!
            gpu_scenes[$gpu_id]="${scene_id}_${sf}_${ef}"

            echo "Assigned scene ${scene_id}[${sf}~${ef}] → GPU-${gpu_id} (PID=${gpu_pids[$gpu_id]})"
            sleep 2
        fi
    done
done

# ── 等待最后一批完成 ──────────────────────────────────────────────────────────
echo "All scenes scheduled (scene_idx=${scene_idx}). Waiting for last batch..."
for gpu_id in "${GPUS[@]}"; do
    [ -n "${gpu_pids[$gpu_id]}" ] && wait "${gpu_pids[$gpu_id]}" && echo "GPU-${gpu_id} done."
done

################################################################################
# 统计
################################################################################
echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
completed=$(wc -l < "${current_dir}/logs/kitti_vde_completed.txt" 2>/dev/null || echo 0)
failed=$(wc -l    < "${current_dir}/logs/kitti_vde_failed.txt"    2>/dev/null || echo 0)
echo "  Completed : ${completed} / ${#scene_defs[@]}"
if [ "${failed}" -gt 0 ]; then
    echo "  Failed    : ${failed}"
    echo "  Failed IDs:"
    cat "${current_dir}/logs/kitti_vde_failed.txt"
fi
echo "  Logs      : ${current_dir}/logs/kitti_vde_train/"
echo "  Outputs   : ${OUTPUT_ROOT}/"
echo "=========================================="