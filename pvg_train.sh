#!/bin/bash

scene_ids=(34)


current_dir=$(pwd)
# 循环处理每个场景
for scene_id in "${scene_ids[@]}"; do

        bash pvg_process.sh "$scene_id" "$current_dir"

done

# 等待所有任务完成
wait

echo "All scenes processed successfully"
