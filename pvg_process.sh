#!/bin/bash

scene_id=$1
current_dir=$2

base_dir=$current_dir
original_waymo_yaml="$base_dir/configs/waymo_nvs.yaml"

# ── 统一 DynamiCrafter checkpoint（改为统一训练后所有场景共用）──
checkpoints_dir="$base_dir/checkpoints/unified_dc/training_512_v1.0_interp/checkpoints"
if [ ! -d "$checkpoints_dir" ]; then
    echo "Error: Unified DC checkpoint not found: $checkpoints_dir"
    exit 1
fi

latest_ckpt=$(find "$checkpoints_dir" -name "*.ckpt" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2)
if [ -z "$latest_ckpt" ]; then
    echo "Error: No .ckpt file found in $checkpoints_dir"
    exit 1
fi

echo "Using checkpoint: $latest_ckpt"

python train.py --config $original_waymo_yaml \
    source_path=$base_dir/data/waymo_scenes/$scene_id \
    model_path=$base_dir/eval_output/waymo_nvs/${scene_id}_end \
    vdm_ckp_dir=$latest_ckpt \
    vdm_config_dir="$base_dir/configs/config_interp_adapt.yaml" \
    vdm_weight=1.0