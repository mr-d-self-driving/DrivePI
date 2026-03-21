#!/bin/bash

#SBATCH --job-name=eval
#SBATCH -p gpu,gpu_shared
#SBATCH --time=96:00:00

#SBATCH --gres=gpu:4
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=30G

source ../../conda_env_cuda12.4.sh drivepi_hwei
nvidia-smi
nvcc -V



torchrun --nproc_per_node=4 --master_port 34321 scripts/emova_batch_inference_occ_with_action.py \
--config configs/example/llava/4dmllm_11_9/drivepi_05b_1e_4.py \
--json_path /grp01/cs_hszhao/zliu12/code/DrivePI_clean/EMOVA_Data/4dmllm_captions/nuscenes_val_only_bev.json \
--output_path /grp01/cs_hszhao/zliu12/code/DrivePI_clean/logdir/drivepi_05b_1e_4.json --batch_size 1 --device cuda



cd occ_evaluation
python eval.py
cd ../

python  evaluate_plan.py
