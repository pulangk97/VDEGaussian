#!/bin/bash
################################################################################
# Modified scene_train.sh for 32 scenes on 8 GPUs
# Stage 1: Scene Adaptation (DynamiCrafter fine-tuning)
################################################################################

scene_ids=(
    "0000001" "0001001" "0002001" "0003001" "0004001" "0005001" "0006001" "0007001" 
    #"0008001" "0009001" "0010001" "0011001" "0012001" "0013001" "0014001" "0015001"
    #"0016001" "0017001" "0018001" "0019001" "0020001" "0021001" "0022001" "0023001"
    #"0024001" "0025001" "0026001" "0027001" "0028001" "0029001" "0030001" "0031001"
)
#GPUS=(8)
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

current_dir=$(pwd)
port_base=12355

mkdir -p "${current_dir}/logs/stage1_adaptation"
> "${current_dir}/logs/stage1_completed.txt"
> "${current_dir}/logs/stage1_failed.txt"

echo "=========================================="
echo "Stage 1: Scene Adaptation"
echo "Scenes: ${#scene_ids[@]}"
echo "GPUs: ${NUM_GPUS}"
echo "=========================================="

################################################################################
# 函数：训练单个场景（不在函数内部做调度，只负责执行）
################################################################################
train_single_scene() {
    local scene_id=$1
    local gpu_id=$2
    local port=$3
    local log_file=$4

    echo "[$(date)] [GPU-${gpu_id}] Starting adaptation for scene ${scene_id}"
    export CUDA_VISIBLE_DEVICES=${gpu_id}

    bash scene_adaptation.sh "${scene_id}" ${port} "${current_dir}" ${gpu_id} \
        >> "${log_file}" 2>&1

    local exit_code=$?

    if [ ${exit_code} -eq 0 ]; then
        echo "[$(date)] [GPU-${gpu_id}] Completed scene ${scene_id}"
        echo "${scene_id}" >> "${current_dir}/logs/stage1_completed.txt"
    else
        echo "[$(date)] [GPU-${gpu_id}] FAILED scene ${scene_id} (exit: ${exit_code})"
        echo "${scene_id}" >> "${current_dir}/logs/stage1_failed.txt"
    fi

    return ${exit_code}
}

################################################################################
# 并行调度逻辑
################################################################################

scene_idx=0
declare -A gpu_pids    # GPU -> PID
declare -A gpu_scenes  # GPU -> scene_id

# 初始化：为每个 GPU 分配第一批场景
for gpu_id in "${GPUS[@]}"; do
    if [ ${scene_idx} -lt ${#scene_ids[@]} ]; then
        scene_id="${scene_ids[$scene_idx]}"
        port=$((port_base + gpu_id))
        log_file="${current_dir}/logs/stage1_adaptation/${scene_id}_gpu${gpu_id}.log"

        train_single_scene "${scene_id}" ${gpu_id} ${port} "${log_file}" &
        # ✅ 关键修复：不使用 local，直接赋值到关联数组
        gpu_pids[$gpu_id]=$!
        gpu_scenes[$gpu_id]="${scene_id}"

        echo "Assigned scene ${scene_id} to GPU-${gpu_id} (PID: ${gpu_pids[$gpu_id]})"
        ((scene_idx++))
        sleep 5
    fi
done

echo "Initial batch launched. scene_idx=${scene_idx}"

# 持续调度剩余场景
while [ ${scene_idx} -lt ${#scene_ids[@]} ]; do
    sleep 30

    for gpu_id in "${GPUS[@]}"; do
        # 跳过没有任务的 GPU（理论上不会出现）
        if [ -z "${gpu_pids[$gpu_id]}" ]; then
            continue
        fi

        # 检查该 GPU 上的进程是否已结束
        if ! kill -0 ${gpu_pids[$gpu_id]} 2>/dev/null; then
            # ✅ 进程已结束，GPU 空闲，分配新场景
            if [ ${scene_idx} -lt ${#scene_ids[@]} ]; then
                scene_id="${scene_ids[$scene_idx]}"
                port=$((port_base + gpu_id))
                log_file="${current_dir}/logs/stage1_adaptation/${scene_id}_gpu${gpu_id}.log"

                echo "GPU-${gpu_id} finished '${gpu_scenes[$gpu_id]}', assigning '${scene_id}' (idx=${scene_idx})"

                train_single_scene "${scene_id}" ${gpu_id} ${port} "${log_file}" &
                # ✅ 关键修复：不使用 local，直接赋值
                gpu_pids[$gpu_id]=$!
                gpu_scenes[$gpu_id]="${scene_id}"

                echo "Assigned scene ${scene_id} to GPU-${gpu_id} (PID: ${gpu_pids[$gpu_id]})"
                ((scene_idx++))
                sleep 2
            fi
        fi
    done
done

# 等待所有剩余任务完成
echo "All ${#scene_ids[@]} scenes scheduled. Waiting for remaining jobs..."
for gpu_id in "${GPUS[@]}"; do
    if [ -n "${gpu_pids[$gpu_id]}" ]; then
        wait ${gpu_pids[$gpu_id]}
        echo "GPU-${gpu_id} final job done."
    fi
done

################################################################################
# 输出统计
################################################################################
echo ""
echo "=========================================="
echo "Stage 1 Completed!"
echo "=========================================="

completed=$(wc -l < "${current_dir}/logs/stage1_completed.txt" 2>/dev/null || echo 0)
failed=$(wc -l < "${current_dir}/logs/stage1_failed.txt" 2>/dev/null || echo 0)

echo "Completed scenes: ${completed}"
if [ "${failed}" -gt 0 ]; then
    echo "Failed scenes: ${failed}"
    cat "${current_dir}/logs/stage1_failed.txt"
fi
echo "=========================================="