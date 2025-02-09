#!/bin/env bash
#SBATCH --array=1
#SBATCH --time=05:00:00
#SBATCH --job-name=3dgs
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100-80g
#SBATCH --mem=32G
#SBATCH --output=scripts/3dgs/log/%A_%a.log
SLURM_ARRAY_TASK_ID=1
# * basic settings
hostname
nvidia-smi

source scripts/tools/basic_config.sh
source scripts/tools/timer_start.sh

# # * data loading, path preparation
path_base=../../../data/GS
output_base=results/3dgs
scene_list=("garden" "bicycle" "stump" "bonsai" "counter" "kitchen" "room" "treehill" "flowers" "drjohnson" "playroom" "train" "truck")
ds_list=("mipnerf360" "mipnerf360" "mipnerf360" "mipnerf360" "mipnerf360" "mipnerf360" "mipnerf360" "mipnerf360" "mipnerf360" "db" "db" "tandt" "tandt")
scene=${scene_list[SLURM_ARRAY_TASK_ID]}
dataset=${ds_list[SLURM_ARRAY_TASK_ID]}
path_source="$path_base"/"$dataset"/"$scene"
path_output="$output_base"/"$dataset"/"$scene"

# # * scripts to run GS model
CUDA_VISIBLE_DEVICES=0 python train.py --eval -s=${path_source} -m=${path_output} 

source scripts/tools/timer_end.sh
# # * rendering
python render.py -m ${output_base}/$dataset/$scene -s=${path_source}  --skip_train 

# # * metrics calculation
python metrics.py -m ${output_base}/$dataset/$scene