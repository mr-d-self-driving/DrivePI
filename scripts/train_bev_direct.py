#!/usr/bin/env python3
"""
Direct training script for BEV features with existing QA data.
No data preparation needed - uses pre-extracted features and QA pairs.
"""

import os
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Direct BEV feature training with existing data")
    parser.add_argument("--config", type=str, 
                       default="configs/example/llava/4dllm/pretrained.py",
                       help="Training configuration file")
    parser.add_argument("--output_dir", type=str,
                       default="./logdir/emova_llava-qwen2_5-3b-bev-feature-pretrain/",
                       help="Output directory for training logs and checkpoints")
    parser.add_argument("--num_epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                       help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Log every N steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm for clipping")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--bf16", action="store_true",
                       help="Use bfloat16 precision")
    parser.add_argument("--tf32", action="store_true",
                       help="Use tf32 precision")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    parser.add_argument("--ddp_timeout", type=int, default=1800,
                       help="DDP timeout in seconds")
    parser.add_argument("--dataloader_pin_memory", action="store_true",
                       help="Pin memory for dataloader")
    parser.add_argument("--remove_unused_columns", action="store_true",
                       help="Remove unused columns from dataset")
    parser.add_argument("--ddp_find_unused_parameters", action="store_true",
                       help="Find unused parameters in DDP")
    parser.add_argument("--ddp_bucket_cap_mb", type=int, default=25,
                       help="DDP bucket cap in MB")
    parser.add_argument("--group_by_length", action="store_true",
                       help="Group sequences by length")
    parser.add_argument("--length_column_name", type=str, default="length",
                       help="Column name for sequence length")
    parser.add_argument("--report_to", type=str, default="none",
                       help="Report to wandb/tensorboard")
    parser.add_argument("--run_name", type=str, default="bev-feature-training",
                       help="Run name for logging")
    parser.add_argument("--max_memory_MB", type=int, default=24000,
                       help="Max memory in MB")
    parser.add_argument("--gradient_checkpointing_kwargs", type=str, default="{}",
                       help="Gradient checkpointing kwargs")
    parser.add_argument("--max_shard_size", type=str, default="5GB",
                       help="Max shard size for model saving")
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=2,
                       help="Dataloader prefetch factor")
    parser.add_argument("--dataloader_persistent_workers", action="store_true",
                       help="Use persistent workers for dataloader")
    parser.add_argument("--optim", type=str, default="adamw_torch",
                       help="Optimizer type")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                       help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                       help="Adam beta2")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                       help="Adam epsilon")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                       help="Weight decay")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                       help="Learning rate scheduler type")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                       help="Warmup ratio")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    print("BEV Feature Direct Training")
    print("=" * 50)
    print(f"Config: {args.config}")
    print(f"Output dir: {args.output_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    
    # Build training command
    cmd_parts = [
        "python", "-m", "emova.train.train",
        f"--config", args.config,
        f"--output_dir", args.output_dir,
        f"--num_train_epochs", str(args.num_epochs),
        f"--per_device_train_batch_size", str(args.batch_size),
        f"--learning_rate", str(args.learning_rate),
        f"--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        f"--warmup_steps", str(args.warmup_steps),
        f"--save_steps", str(args.save_steps),
        f"--eval_steps", str(args.eval_steps),
        f"--logging_steps", str(args.logging_steps),
        f"--max_grad_norm", str(args.max_grad_norm),
        f"--dataloader_num_workers", str(args.dataloader_num_workers),
        f"--ddp_timeout", str(args.ddp_timeout),
        f"--report_to", args.report_to,
        f"--run_name", args.run_name,
        f"--max_memory_MB", str(args.max_memory_MB),
        f"--max_shard_size", args.max_shard_size,
        f"--dataloader_prefetch_factor", str(args.dataloader_prefetch_factor),
        f"--optim", args.optim,
        f"--adam_beta1", str(args.adam_beta1),
        f"--adam_beta2", str(args.adam_beta2),
        f"--adam_epsilon", str(args.adam_epsilon),
        f"--weight_decay", str(args.weight_decay),
        f"--lr_scheduler_type", args.lr_scheduler_type,
        f"--warmup_ratio", str(args.warmup_ratio),
    ]
    
    # Add optional flags
    if args.bf16:
        cmd_parts.append("--bf16")
    if args.tf32:
        cmd_parts.append("--tf32")
    if args.gradient_checkpointing:
        cmd_parts.append("--gradient_checkpointing")
    if args.dataloader_pin_memory:
        cmd_parts.append("--dataloader_pin_memory")
    if args.remove_unused_columns:
        cmd_parts.append("--remove_unused_columns")
    if args.ddp_find_unused_parameters:
        cmd_parts.append("--ddp_find_unused_parameters")
    if args.group_by_length:
        cmd_parts.append("--group_by_length")
    if args.dataloader_persistent_workers:
        cmd_parts.append("--dataloader_persistent_workers")
    if args.resume_from_checkpoint:
        cmd_parts.extend(["--resume_from_checkpoint", args.resume_from_checkpoint])
    if args.local_rank != -1:
        cmd_parts.extend(["--local_rank", str(args.local_rank)])
    
    # Add gradient checkpointing kwargs if specified
    if args.gradient_checkpointing_kwargs != "{}":
        cmd_parts.extend(["--gradient_checkpointing_kwargs", args.gradient_checkpointing_kwargs])
    
    # Add length column name
    cmd_parts.extend(["--length_column_name", args.length_column_name])
    
    # Add DDP bucket cap
    cmd_parts.extend(["--ddp_bucket_cap_mb", str(args.ddp_bucket_cap_mb)])
    
    cmd = " ".join(cmd_parts)
    
    print(f"\nTraining command:")
    print(f"{cmd}")
    print(f"\nStarting training...")
    
    # Execute training
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print("Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 