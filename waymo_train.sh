#!/bin/bash

scene_ids=(
    0000001 
    #0001001 0002001 0003001 0004001 0005001 0006001 0007001
    #0008001 0009001 0010001 0011001 0012001 0013001 0014001 0015001
    #0016001 0017001 0018001 0019001 0020001 0021001 0022001 0023001
    #0024001 0025001 0026001 0027001 0028001 0029001 0030001 0031001
)
GPUS=(0)
NUM_GPUS=${#GPUS[@]}

current_dir=$(pwd)

mkdir -p "${current_dir}/logs/stage2_training"
> "${current_dir}/logs/stage2_completed.txt"
> "${current_dir}/logs/stage2_failed.txt"
> "${current_dir}/logs/stage2_skipped.txt"

UNIFIED_DC_CKPT="${current_dir}/checkpoints/unified_dc/training_512_v1.0_interp/checkpoints"

echo "=========================================="
echo "Stage 2: PVG Training (4D Gaussian Splatting)"
echo "Scenes:     ${#scene_ids[@]}"
echo "GPUs:       ${NUM_GPUS}"
echo "DC ckpt:    ${UNIFIED_DC_CKPT}"
echo "=========================================="


if [ ! -d "${UNIFIED_DC_CKPT}" ]; then
    echo "Error: Unified DynamiCrafter checkpoint not found at ${UNIFIED_DC_CKPT}"
    echo "Please run train_unified_dc.sh first."
    exit 1
fi


train_single_scene() {
    local scene_id=$1
    local gpu_id=$2
    local log_file=$3

    echo "[$(date)] [GPU-${gpu_id}] Starting PVG training for scene ${scene_id}"
    export CUDA_VISIBLE_DEVICES=${gpu_id}

    bash pvg_process.sh ${scene_id} ${current_dir} >> "${log_file}" 2>&1

    local exit_code=$?
    if [ ${exit_code} -eq 0 ]; then
        echo "[$(date)] [GPU-${gpu_id}] Completed scene ${scene_id}"
        echo "${scene_id}" >> "${current_dir}/logs/stage2_completed.txt"
    else
        echo "[$(date)] [GPU-${gpu_id}] FAILED scene ${scene_id} (exit: ${exit_code})"
        echo "${scene_id}" >> "${current_dir}/logs/stage2_failed.txt"
    fi
    return ${exit_code}
}


scene_idx=0
declare -A gpu_pids
declare -A gpu_scenes

for gpu_id in "${GPUS[@]}"; do
    if [ ${scene_idx} -lt ${#scene_ids[@]} ]; then
        scene_id="${scene_ids[$scene_idx]}"
        ((scene_idx++))
        log_file="${current_dir}/logs/stage2_training/${scene_id}_gpu${gpu_id}.log"
        train_single_scene "${scene_id}" ${gpu_id} "${log_file}" &
        gpu_pids[$gpu_id]=$!
        gpu_scenes[$gpu_id]="${scene_id}"
        echo "Assigned scene ${scene_id} to GPU-${gpu_id} (PID: ${gpu_pids[$gpu_id]})"
        sleep 5
    fi
done

echo "Initial batch launched. scene_idx=${scene_idx} / ${#scene_ids[@]}"

while [ ${scene_idx} -lt ${#scene_ids[@]} ]; do
    sleep 30
    for gpu_id in "${GPUS[@]}"; do
        [ -z "${gpu_pids[$gpu_id]}" ] && continue
        [ ${scene_idx} -ge ${#scene_ids[@]} ] && break

        if ! kill -0 "${gpu_pids[$gpu_id]}" 2>/dev/null; then
            scene_id="${scene_ids[$scene_idx]}"
            ((scene_idx++))
            log_file="${current_dir}/logs/stage2_training/${scene_id}_gpu${gpu_id}.log"
            echo "GPU-${gpu_id} finished '${gpu_scenes[$gpu_id]}', assigning '${scene_id}' (scene_idx=${scene_idx})"
            train_single_scene "${scene_id}" ${gpu_id} "${log_file}" &
            gpu_pids[$gpu_id]=$!
            gpu_scenes[$gpu_id]="${scene_id}"
            echo "Assigned scene ${scene_id} to GPU-${gpu_id} (PID: ${gpu_pids[$gpu_id]})"
            sleep 2
        fi
    done
done

echo "All scenes scheduled. Waiting for remaining jobs..."
for gpu_id in "${GPUS[@]}"; do
    if [ -n "${gpu_pids[$gpu_id]}" ]; then
        wait "${gpu_pids[$gpu_id]}"
        echo "GPU-${gpu_id} final job done."
    fi
done

echo ""
echo "=========================================="
echo "Stage 2 Completed!"
echo "=========================================="
completed=$(wc -l < "${current_dir}/logs/stage2_completed.txt" 2>/dev/null || echo 0)
failed=$(wc -l    < "${current_dir}/logs/stage2_failed.txt"    2>/dev/null || echo 0)
skipped=$(wc -l   < "${current_dir}/logs/stage2_skipped.txt"   2>/dev/null || echo 0)
echo "Completed : ${completed}"
[ "${skipped}" -gt 0 ] && echo "Skipped   : ${skipped}" && cat "${current_dir}/logs/stage2_skipped.txt"
[ "${failed}"  -gt 0 ] && echo "Failed    : ${failed}"  && cat "${current_dir}/logs/stage2_failed.txt"
echo "=========================================="