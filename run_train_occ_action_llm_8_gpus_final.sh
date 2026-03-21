#!/bin/bash

#SBATCH --job-name=tre_v1
#SBATCH -p gpu,gpu_shared
#SBATCH --time=96:00:00

#SBATCH --gres=gpu:4
#SBATCH -N 1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-gpu=48G

source ../../conda_env_cuda12.4.sh drivepi_hwei
nvidia-smi
nvcc -V


# torchrun --nproc_per_node=4 --master_port 52281 emova/train/train.py configs/example/drivepi/pretrain_sms_0722_three_rh_05b.py


torchrun --nproc_per_node=4 --master_port 52281 emova/train/train.py configs/example/drivepi/drivepi_05b_1e_4.py



