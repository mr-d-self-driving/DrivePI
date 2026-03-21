#!/bin/bash


conda activate drivepi
nvidia-smi
nvcc -V


# torchrun --nproc_per_node=4 --master_port 52281 emova/train/train.py configs/example/drivepi/pretrain_sms_0722_three_rh_05b.py


torchrun --nproc_per_node=4 --master_port 52281 emova/train/train.py configs/example/drivepi/drivepi_05b_1e_4.py



