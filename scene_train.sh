#!/bin/bash

# 定义场景ID列表
# scene_ids=(16 21 22 25 31 34 35 49 53 80 84 86 89 94 96 102 111 222 323 382 402 427 438 546 581 592 620 640 700 754 795 796)
scene_ids=(34)
current_dir=$(pwd)
echo $current_dir

port_base=12355
task_index=0




# 循环处理每个场景
for scene_id in "${scene_ids[@]}"; do

        bash scene_adaptation.sh "$scene_id"  $((port_base + task_index)) "$current_dir"
        task_index=$((task_index + 1))

done


echo "All scenes processed successfully"
