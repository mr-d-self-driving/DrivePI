#!/bin/bash

# BEV Feature 训练启动脚本
# 适用于已有 BEV 特征和 QA 数据的直接训练

echo "=== BEV Feature 训练启动 ==="
echo "时间: $(date)"
echo "工作目录: $(pwd)"
echo ""

# 检查数据文件是否存在
echo "检查数据文件..."
if [ ! -f "/data/zliu/code/hw_proj/nuscenes_infos_train.pkl" ]; then
    echo "错误: nuScenes 元数据文件不存在"
    exit 1
fi

if [ ! -f "/data/runhui/captions/nuscenes_train_28130_caption.json" ]; then
    echo "错误: QA 数据文件不存在"
    exit 1
fi

if [ ! -d "/data/zliu/code/hw_proj/" ]; then
    echo "错误: BEV 特征目录不存在"
    exit 1
fi

echo "✓ 数据文件检查通过"
echo ""

# 设置默认参数
OUTPUT_DIR="./logdir/emova_llava-qwen2_5-3b-bev-feature-pretrain/"
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=16
LEARNING_RATE=2e-5
NUM_EPOCHS=1

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gradient_accumulation_steps)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --bf16)
            BF16_FLAG="--bf16"
            shift
            ;;
        --gradient_checkpointing)
            GRADIENT_CHECKPOINTING_FLAG="--gradient_checkpointing"
            shift
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --output_dir DIR                   输出目录 (默认: $OUTPUT_DIR)"
            echo "  --batch_size SIZE                  批次大小 (默认: $BATCH_SIZE)"
            echo "  --gradient_accumulation_steps STEPS 梯度累积步数 (默认: $GRADIENT_ACCUMULATION_STEPS)"
            echo "  --learning_rate LR                 学习率 (默认: $LEARNING_RATE)"
            echo "  --num_epochs EPOCHS                训练轮数 (默认: $NUM_EPOCHS)"
            echo "  --bf16                             使用 bfloat16 精度"
            echo "  --gradient_checkpointing           启用梯度检查点"
            echo "  --help                             显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 显示训练配置
echo "训练配置:"
echo "  输出目录: $OUTPUT_DIR"
echo "  批次大小: $BATCH_SIZE"
echo "  梯度累积步数: $GRADIENT_ACCUMULATION_STEPS"
echo "  学习率: $LEARNING_RATE"
echo "  训练轮数: $NUM_EPOCHS"
echo "  bf16: ${BF16_FLAG:-未启用}"
echo "  梯度检查点: ${GRADIENT_CHECKPOINTING_FLAG:-未启用}"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 启动训练
echo "启动训练..."
python scripts/train_bev_direct.py \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    $BF16_FLAG \
    $GRADIENT_CHECKPOINTING_FLAG

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 训练完成!"
    echo "输出目录: $OUTPUT_DIR"
else
    echo ""
    echo "✗ 训练失败!"
    exit 1
fi 