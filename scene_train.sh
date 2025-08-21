#!/bin/bash

scene_ids=(16)
current_dir=$(pwd)
echo $current_dir

port_base=12355
task_index=0


for scene_id in "${scene_ids[@]}"; do

        bash scene_adaptation.sh "$scene_id"  $((port_base + task_index)) "$current_dir"
        task_index=$((task_index + 1))

done


echo "All scenes processed successfully"
