#!/bin/bash



scene_id=$1
current_port=$2
current_dir=$3

base_dir=$current_dir
dc_dir="$base_dir/submodules/DynamiCrafter"
original_dc_config="$base_dir/configs/config_interp_adapt.yaml"
original_base_yaml="$base_dir/configs/base.yaml"
original_waymo_yaml="$base_dir/configs/waymo_nvs.yaml"
trainer_sh="$base_dir/run_interp.sh"


temp_dir=$(mktemp -d)
trap 'rm -rf "$temp_dir"' EXIT INT TERM


for file in "$original_dc_config" "$trainer_sh" "$original_base_yaml" "$original_waymo_yaml"; do
    if [ ! -f "$file" ]; then
        echo "Error: Required file not found at $file"
        exit 1
    fi
done


temp_dc_config="$temp_dir/config_interp_adapt.yaml"
temp_base_yaml="$temp_dir/base.yaml"
temp_waymo_yaml="$temp_dir/waymo_nvs.yaml"

cp "$original_dc_config" "$temp_dc_config" || exit 1
cp "$original_base_yaml" "$temp_base_yaml" || exit 1
cp "$original_waymo_yaml" "$temp_waymo_yaml" || exit 1

# 修改临时配置文件
sed -i "s|\(data_dir: \)\"./data/waymo_scenes/16\"|\1\"$base_dir/data/waymo_scenes/$scene_id\"|g" "$temp_dc_config" || exit 1

# 执行训练脚本
if ! bash "$trainer_sh" "$scene_id" "$temp_dc_config" "$current_port" "$base_dir"; then
    echo "Error: Failed to execute training script $trainer_sh"
    exit 1
fi

