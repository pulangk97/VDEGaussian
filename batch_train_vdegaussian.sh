
#!/bin/bash
################################################################################
# VDEGaussian 32场景批量训练脚本
# 支持7张4090显卡并行训练
# 基于VDEGaussian的两阶段训练流程：
#   Stage 1: Scene Adaptation (DynamiCrafter微调)
#   Stage 2: 4DGS Training (重建模型训练)
################################################################################

set -e  # 遇到错误立即退出

################################################################################
# 配置区域
################################################################################

# 32个Waymo场景ID（EmerNeRF NOTR Static-32数据集）
SCENES=(
    "0009001" "0010001" "0011001"
    "0012001" "0013001" "0014001" "0015001" "0016001" "0017001" "0018001" "0019001"
    "0020001" "0021001" "0022001" "0023001" "0024001" "0025001" "0026001" "0027001"
    "0028001" "0029001" "0030001" "0031001" 
)

# GPU配置
NUM_GPUS=7
GPU_IDS=(0 1 2 3 4 5 6)

# 路径配置
DATA_ROOT="data/waymo_scenes"
OUTPUT_ROOT="./output/waymo_32scenes"
CONFIG_FILE="configs/waymo_nvs.yaml"

# 自定义数据划分参数
TEST_SAMPLING_STRATEGY="custom"  # custom / fixed / original
TEST_EVERY=10
TEST_NUM_PER_BLOCK=2
SPLIT_SEED=42

# 训练参数
ITERATIONS=30000
PORT_START=6009

# Stage控制（可以选择只运行某个阶段）
RUN_STAGE1=true   # Scene Adaptation
RUN_STAGE2=true   # 4DGS Training
RUN_EVAL=true     # 评估

################################################################################
# 辅助函数
################################################################################

# 日志函数
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] $1"
}

# 检查GPU是否空闲
is_gpu_idle() {
    local gpu_id=$1
    # 检查该GPU上是否有Python训练进程
    if ps aux | grep -v grep | grep "CUDA_VISIBLE_DEVICES=${gpu_id}" | grep "python.*train.py" > /dev/null; then
        return 1  # GPU忙碌
    else
        return 0  # GPU空闲
    fi
}

# 等待任意一个GPU空闲
wait_for_any_gpu() {
    while true; do
        for gpu_id in "${GPU_IDS[@]}"; do
            if is_gpu_idle ${gpu_id}; then
                echo ${gpu_id}
                return 0
            fi
        done
        sleep 30  # 等待30秒后重新检查
    done
}

# 等待所有GPU完成
wait_for_all_gpus() {
    log_info "Waiting for all GPUs to finish..."
    while true; do
        all_idle=true
        for gpu_id in "${GPU_IDS[@]}"; do
            if ! is_gpu_idle ${gpu_id}; then
                all_idle=false
                break
            fi
        done
        
        if ${all_idle}; then
            break
        fi
        sleep 30
    done
    log_success "All GPUs finished!"
}

################################################################################
# Stage 1: Scene Adaptation (DynamiCrafter微调)
################################################################################

scene_adaptation() {
    local scene=$1
    local gpu_id=$2
    local log_file=$3
    
    log_info "[GPU-${gpu_id}] Stage 1: Scene Adaptation for ${scene}"
    
    local scene_dir="${DATA_ROOT}/${scene}"
    local output_dir="${OUTPUT_ROOT}/stage1_adaptation/${scene}"
    
    # 检查数据是否存在
    if [ ! -d "${scene_dir}" ]; then
        log_error "Scene directory not found: ${scene_dir}"
        return 1
    fi
    
    # 创建输出目录
    mkdir -p ${output_dir}
    
    # 运行Scene Adaptation
    CUDA_VISIBLE_DEVICES=${gpu_id} bash scene_adaptation.sh \
        --source_path ${scene_dir} \
        --model_path ${output_dir} \
        --test_sampling_strategy ${TEST_SAMPLING_STRATEGY} \
        --test_every ${TEST_EVERY} \
        --test_num_per_block ${TEST_NUM_PER_BLOCK} \
        --split_seed ${SPLIT_SEED} \
        >> ${log_file} 2>&1
    
    local exit_code=$?
    
    if [ ${exit_code} -eq 0 ]; then
        log_success "[GPU-${gpu_id}] Stage 1 completed: ${scene}"
        return 0
    else
        log_error "[GPU-${gpu_id}] Stage 1 failed: ${scene} (exit code: ${exit_code})"
        return 1
    fi
}

################################################################################
# Stage 2: 4DGS Training (重建模型训练)
################################################################################

train_4dgs() {
    local scene=$1
    local gpu_id=$2
    local port=$3
    local log_file=$4
    
    log_info "[GPU-${gpu_id}] Stage 2: 4DGS Training for ${scene}"
    
    local scene_dir="${DATA_ROOT}/${scene}"
    local output_dir="${OUTPUT_ROOT}/stage2_reconstruction/${scene}"
    local adaptation_dir="${OUTPUT_ROOT}/stage1_adaptation/${scene}"
    
    # 创建输出目录
    mkdir -p ${output_dir}
    
    # 运行4DGS训练
    CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
        -s ${scene_dir} \
        -m ${output_dir} \
        --config ${CONFIG_FILE} \
        --test_sampling_strategy ${TEST_SAMPLING_STRATEGY} \
        --test_every ${TEST_EVERY} \
        --test_num_per_block ${TEST_NUM_PER_BLOCK} \
        --split_seed ${SPLIT_SEED} \
        --iterations ${ITERATIONS} \
        --port ${port} \
        --eval \
        --adaptation_path ${adaptation_dir} \
        >> ${log_file} 2>&1
    
    local exit_code=$?
    
    if [ ${exit_code} -eq 0 ]; then
        log_success "[GPU-${gpu_id}] Stage 2 completed: ${scene}"
        return 0
    else
        log_error "[GPU-${gpu_id}] Stage 2 failed: ${scene} (exit code: ${exit_code})"
        return 1
    fi
}

################################################################################
# 评估函数
################################################################################

evaluate_scene() {
    local scene=$1
    local gpu_id=$2
    local log_file=$3
    
    log_info "[GPU-${gpu_id}] Evaluating ${scene}"
    
    local model_path="${OUTPUT_ROOT}/stage2_reconstruction/${scene}"
    
    CUDA_VISIBLE_DEVICES=${gpu_id} python evaluate.py \
        -m ${model_path} \
        >> ${log_file} 2>&1
    
    local exit_code=$?
    
    if [ ${exit_code} -eq 0 ]; then
        log_success "[GPU-${gpu_id}] Evaluation completed: ${scene}"
        return 0
    else
        log_error "[GPU-${gpu_id}] Evaluation failed: ${scene}"
        return 1
    fi
}

################################################################################
# 主训练函数（完整pipeline）
################################################################################

train_scene_pipeline() {
    local scene=$1
    local gpu_id=$2
    local port=$3
    
    local log_dir="${OUTPUT_ROOT}/logs"
    mkdir -p ${log_dir}
    
    local log_file="${log_dir}/${scene}_gpu${gpu_id}.log"
    local status_file="${log_dir}/${scene}_status.txt"
    
    log_info "[GPU-${gpu_id}] Starting pipeline for ${scene}"
    echo "STARTED" > ${status_file}
    
    # Stage 1: Scene Adaptation
    if ${RUN_STAGE1}; then
        if ! scene_adaptation ${scene} ${gpu_id} ${log_file}; then
            echo "STAGE1_FAILED" > ${status_file}
            return 1
        fi
        echo "STAGE1_DONE" > ${status_file}
    fi
    
    # Stage 2: 4DGS Training
    if ${RUN_STAGE2}; then
        if ! train_4dgs ${scene} ${gpu_id} ${port} ${log_file}; then
            echo "STAGE2_FAILED" > ${status_file}
            return 1
        fi
        echo "STAGE2_DONE" > ${status_file}
    fi
    
    # Evaluation
    if ${RUN_EVAL}; then
        if ! evaluate_scene ${scene} ${gpu_id} ${log_file}; then
            echo "EVAL_FAILED" > ${status_file}
            return 1
        fi
        echo "COMPLETED" > ${status_file}
    fi
    
    log_success "[GPU-${gpu_id}] Pipeline completed for ${scene}"
    return 0
}

################################################################################
# 并行调度器
################################################################################

run_parallel_training() {
    log_info "Starting parallel training on ${NUM_GPUS} GPUs"
    log_info "Total scenes: ${#SCENES[@]}"
    
    local pids=()
    local gpu_map=()  # 记录每个进程使用的GPU
    local scene_map=()  # 记录每个进程处理的场景
    
    for scene in "${SCENES[@]}"; do
        # 等待一个空闲GPU
        log_info "Waiting for available GPU for scene ${scene}..."
        local gpu_id=$(wait_for_any_gpu)
        local port=$((PORT_START + gpu_id))
        
        log_info "Assigning scene ${scene} to GPU ${gpu_id}"
        
        # 后台运行训练
        train_scene_pipeline ${scene} ${gpu_id} ${port} &
        local pid=$!
        
        pids+=($pid)
        gpu_map+=($gpu_id)
        scene_map+=(${scene})
        
        # 短暂延迟，避免同时启动过多进程
        sleep 5
    done
    
    # 等待所有任务完成
    log_info "All scenes scheduled. Waiting for completion..."
    
    local failed_scenes=()
    for i in "${!pids[@]}"; do
        local pid=${pids[$i]}
        local scene=${scene_map[$i]}
        local gpu_id=${gpu_map[$i]}
        
        wait $pid
        local exit_code=$?
        
        if [ ${exit_code} -ne 0 ]; then
            log_error "Scene ${scene} failed on GPU ${gpu_id}"
            failed_scenes+=(${scene})
        else
            log_success "Scene ${scene} completed successfully on GPU ${gpu_id}"
        fi
    done
    
    # 报告失败的场景
    if [ ${#failed_scenes[@]} -gt 0 ]; then
        log_error "Failed scenes: ${failed_scenes[@]}"
        return 1
    else
        log_success "All scenes completed successfully!"
        return 0
    fi
}

################################################################################
# 结果汇总
################################################################################

collect_results() {
    log_info "Collecting results..."
    
    local results_file="${OUTPUT_ROOT}/results_summary.txt"
    local status_dir="${OUTPUT_ROOT}/logs"
    
    echo "================================" > ${results_file}
    echo "VDEGaussian 32 Scenes Training" >> ${results_file}
    echo "Date: $(date)" >> ${results_file}
    echo "================================" >> ${results_file}
    echo "" >> ${results_file}
    
    local completed=0
    local failed=0
    
    for scene in "${SCENES[@]}"; do
        local status_file="${status_dir}/${scene}_status.txt"
        
        if [ -f ${status_file} ]; then
            local status=$(cat ${status_file})
            echo "Scene ${scene}: ${status}" >> ${results_file}
            
            if [ "${status}" == "COMPLETED" ]; then
                ((completed++))
            else
                ((failed++))
            fi
        else
            echo "Scene ${scene}: NOT_STARTED" >> ${results_file}
            ((failed++))
        fi
    done
    
    echo "" >> ${results_file}
    echo "Summary:" >> ${results_file}
    echo "  Completed: ${completed}" >> ${results_file}
    echo "  Failed: ${failed}" >> ${results_file}
    echo "  Total: ${#SCENES[@]}" >> ${results_file}
    
    log_info "Results saved to ${results_file}"
    cat ${results_file}
}

################################################################################
# 主程序
################################################################################

main() {
    echo "========================================================================"
    echo "         VDEGaussian 32 Scenes Batch Training Pipeline"
    echo "========================================================================"
    echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Scenes: ${#SCENES[@]}"
    echo "GPUs: ${NUM_GPUS} (IDs: ${GPU_IDS[@]})"
    echo "Strategy: ${TEST_SAMPLING_STRATEGY}"
    echo "Test sampling: Every ${TEST_EVERY} frames, sample ${TEST_NUM_PER_BLOCK}"
    echo "========================================================================"
    echo ""
    
    # 检查必要的目录和文件
    if [ ! -d "${DATA_ROOT}" ]; then
        log_error "Data root directory not found: ${DATA_ROOT}"
        exit 1
    fi
    
    if [ ! -f "${CONFIG_FILE}" ]; then
        log_error "Config file not found: ${CONFIG_FILE}"
        exit 1
    fi
    
    # 创建输出目录
    mkdir -p ${OUTPUT_ROOT}/logs
    mkdir -p ${OUTPUT_ROOT}/stage1_adaptation
    mkdir -p ${OUTPUT_ROOT}/stage2_reconstruction
    
    # 记录开始时间
    local start_time=$(date +%s)
    
    # 运行并行训练
    run_parallel_training
    local train_exit_code=$?
    
    # 记录结束时间
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    
    log_info "Total training time: ${hours}h ${minutes}m"
    
    # 汇总结果
    collect_results
    
    # 返回状态
    if [ ${train_exit_code} -eq 0 ]; then
        log_success "All training completed successfully!"
        exit 0
    else
        log_error "Some scenes failed. Check logs for details."
        exit 1
    fi
}

# 运行主程序
main "$@"