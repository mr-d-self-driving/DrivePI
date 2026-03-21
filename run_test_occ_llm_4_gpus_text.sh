#!/bin/bash


conda activate drivepi
nvidia-smi
nvcc -V


torchrun --nproc_per_node=4 --master_port 53298 scripts/eval_qa.py \
 --config configs/example/drivepi/drivepi_05b_1e_4.py \
 --batch_size 1 \
 --device cuda
