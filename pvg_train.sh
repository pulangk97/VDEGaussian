#!/bin/bash

scene_ids=(16)


current_dir=$(pwd)
for scene_id in "${scene_ids[@]}"; do

        bash pvg_process.sh "$scene_id" "$current_dir"

done

wait

echo "All scenes processed successfully"
