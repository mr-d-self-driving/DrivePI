#!/bin/bash

conda activate drivepi

torchrun --nproc_per_node=4 --master_port 34321 scripts/emova_batch_inference_occ_with_action.py \
--config configs/example/drivepi/drivepi_05b_1e_4.py \
--json_path /path/DrivePI_Data/drivepi_captions/nuscenes_val_only_bev.json \
--output_path /path/logdir/drivepi_05b_1e_4.json --batch_size 1 --device cuda



#cd occ_evaluation
#python eval.py
#cd ../
#
#python  evaluate_plan.py
