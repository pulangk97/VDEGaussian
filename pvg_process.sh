#!/bin/bash


scene_id=$1
current_dir=$2

base_dir=$current_dir
dc_dir="$base_dir/submodules/DynamiCrafter"
original_dc_config="$base_dir/configs/config_interp_adapt.yaml"
original_base_yaml="$base_dir/configs/base.yaml"
original_waymo_yaml="$base_dir/configs/waymo_nvs.yaml"
trainer_sh="$base_dir/run_interp.sh"



checkpoints_dir="$base_dir/checkpoints/waymo_adapt_testhold_$scene_id/training_512_v1.0_interp/checkpoints"
if [ ! -d "$checkpoints_dir" ]; then
    echo "Error: Checkpoints directory not found: $checkpoints_dir"
    exit 1
fi

latest_ckpt=$(find "$checkpoints_dir" -name "*.ckpt" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2)
if [ -z "$latest_ckpt" ]; then
    echo "Error: No .ckpt file found in $checkpoints_dir"
    exit 1
fi

echo "Using checkpoint file: $latest_ckpt"


python train.py --config $original_waymo_yaml \
    source_path=$base_dir/data/waymo_scenes/$scene_id \
    model_path=$base_dir/eval_output/waymo_nvs/${scene_id}_end \
    vdm_ckp_dir=$latest_ckpt \
    vdm_config_dir="$base_dir/configs/config_interp_adapt.yaml" \
    vdm_weight=1.0
