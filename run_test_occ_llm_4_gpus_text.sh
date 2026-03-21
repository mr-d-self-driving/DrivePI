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


torchrun --nproc_per_node=4 --master_port 53298 scripts/eval_qa.py \
 --config configs/example/drivepi/drivepi_05b_1e_4.py \
 --batch_size 1 \
 --device cuda
