
scene_id=$1
master_port=$3
name="training_512_v1.0"
config_file=$2
base_dir=$4

# save root dir for logs, checkpoints, tensorboard record, etc.
save_root="$base_dir/submodules/DynamiCrafter/results/waymo_adapt_testhold_$scene_id"
mkdir -p $save_root/${name}_interp

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=$master_port --node_rank=0 \
./submodules/DynamiCrafter/main/trainer.py \
--base $config_file \
--train \
--name ${name}_interp \
--logdir $save_root \
--devices 1 \
lightning.trainer.num_nodes=1
