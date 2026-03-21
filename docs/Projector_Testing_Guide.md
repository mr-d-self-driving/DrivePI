# EMOVA Projector 测试指南

本指南介绍如何使用修改后的推理脚本来测试LLaVA第一阶段的projector结果。

## 概述

在LLaVA的训练过程中，第一阶段主要训练vision tower和projector，第二阶段才训练语言模型。为了测试第一阶段的训练效果，我们修改了推理脚本，支持按照训练时的初始化方法初始化模型，然后只给projector加载训练好的权重。

**重要说明**: `ProjectorOnlyChatbot` 和 `Chatbot` 的区别只在模型初始化部分。`ProjectorOnlyChatbot` 按照训练时的初始化方法初始化模型，然后只给projector加载训练好的权重，但实际使用时都支持完整的QA功能。

## 修改内容

### 1. 新增参数

- `--projector_only`: 启用projector-only模式，使用训练时的初始化方法
- `--projector_output_path`: 指定输出文件的保存路径
- `--projector_weights_path`: 指定训练好的projector权重文件路径（可选）

### 2. 新增类

- `ProjectorOnlyChatbot`: 按照训练时的初始化方法初始化模型，然后只给projector加载训练好的权重，支持完整的QA功能

## 使用方法

### 1. 使用ProjectorOnlyChatbot进行完整QA推理

按照训练时的初始化方法初始化模型，进行完整的QA推理：

```bash
python EMOVA/scripts/emova_batch_inference_projector.py \
    --config your_config_file.py \
    --json_path test_data.json \
    --projector_output_path qa_results.json \
    --projector_only \
    --batch_size 2 \
    --verbose
```

### 2. 使用ProjectorOnlyChatbot输出projector特征（调试用）

如果输出文件名包含"projector"，则只输出projector特征用于调试：

```bash
python EMOVA/scripts/emova_batch_inference_projector.py \
    --config your_config_file.py \
    --json_path test_data.json \
    --projector_output_path projector_features.json \
    --projector_only \
    --batch_size 2 \
    --verbose
```

### 3. 指定projector权重路径

```bash
python EMOVA/scripts/emova_batch_inference_projector.py \
    --config your_config_file.py \
    --json_path test_data.json \
    --projector_output_path qa_results.json \
    --projector_weights_path /path/to/your/mm_projector.bin \
    --projector_only \
    --batch_size 2 \
    --verbose
```

### 4. 使用原始Chatbot（完整模型模式）

```bash
python EMOVA/scripts/emova_batch_inference_projector.py \
    --config your_config_file.py \
    --json_path test_data.json \
    --output_path full_model_results.json \
    --batch_size 2 \
    --verbose
```

**输出格式：**

QA推理输出：
```json
[
    "The image shows a cat sitting on a windowsill.",
    "There are several cars parked in the parking lot."
]
```

Projector特征输出（调试用）：
```json
[
    {
        "projected_features": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
        "feature_shape": [256, 4096],
        "original_image_size": [224, 224]
    }
]
```

## 初始化方法

### 训练时的初始化流程

1. **构建语言模型**: 使用`build_language_model`按照配置初始化语言模型
2. **初始化vision modules**: 调用`model.initialize_vision_modules(model_args)`初始化vision tower和projector
3. **加载projector权重**: 从指定的权重文件加载训练好的projector权重

### 关键特点

- **完整初始化**: 按照训练时的完整流程初始化模型
- **权重加载**: 只给projector加载训练好的权重，其他组件保持初始化状态
- **灵活配置**: 支持指定权重文件路径或使用默认路径

## 配置文件要求

确保你的配置文件中包含正确的projector设置：

```python
# configs/your_config.py
model_args = dict(
    version="llama3",
    
    # Vision Tower 配置
    mm_vision_tower=dict(
        type='CLIPVisionTower',
        pretrained_model_name_or_path='openai/clip-vit-large-patch14-336',
        mm_vision_select_layer=-2,
        mm_vision_select_feature='patch',
    ),
    
    # Projector 配置
    mm_projector=dict(
        type='MLPProjector',  # 或其他projector类型
        mlp_depth=2,
    ),
    
    # 语言模型配置
    language_model=dict(
        type='EmovaLlamaForCausalLM',
        pretrained_model_name_or_path='meta-llama/Llama-3.1-8B-Instruct',
        from_pretrained=True,
    ),
)

training_args = dict(
    output_dir="./your_model_output_dir",  # 模型输出目录
)
```

## 支持的Projector类型

EMOVA支持多种projector类型：

### 1. MLPProjector
最简单的MLP投影器：
```python
mm_projector=dict(
    type='MLPProjector',
    mlp_depth=2,
)
```

### 2. CAbstractorMMProjector
带卷积的抽象器：
```python
mm_projector=dict(
    type='CAbstractorMMProjector',
    conv_block_depth=2,
    downsample_rate=4,
    downsample_size=(16, 16),
    num_input_token=1024,
    add_pos_embed=False
)
```

### 3. SelfMiningMMProjector
自挖掘投影器：
```python
mm_projector=dict(
    type='SelfMiningMMProjector',
    conv_block_depth=2,
    downsample_rate=4,
    downsample_size=(16, 16),
    num_input_token=1024,
)
```

### 4. ConcatChannelMMProjector
通道拼接投影器：
```python
mm_projector=dict(
    type='ConcatChannelMMProjector',
    downsample_rate=3,
    downsample_size=(60, 60),
    num_input_token=180 * 180,
    mlp_depth=2,
)
```

## 测试数据格式

输入JSON文件格式：
```json
[
    {
        "image": "path/to/image1.jpg",
        "question": "Describe what you see in this image."
    },
    {
        "image": "path/to/image2.jpg", 
        "question": "What objects are visible?"
    }
]
```

## 分析Projector输出

### 1. 特征统计
```python
import json
import numpy as np

with open("projector_results.json", "r") as f:
    results = json.load(f)

for i, result in enumerate(results):
    features = np.array(result['projected_features'])
    print(f"样本 {i}:")
    print(f"  特征形状: {result['feature_shape']}")
    print(f"  均值: {features.mean():.4f}")
    print(f"  标准差: {features.std():.4f}")
    print(f"  范围: [{features.min():.4f}, {features.max():.4f}]")
```

### 2. 特征可视化
```python
import matplotlib.pyplot as plt

# 可视化特征分布
features = np.array(results[0]['projected_features'])
plt.figure(figsize=(10, 6))
plt.hist(features.flatten(), bins=50, alpha=0.7)
plt.title('Projected Features Distribution')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.show()
```

## 性能优化建议

### 1. 内存优化
- 使用较小的batch_size
- 设置合适的torch_dtype（fp16或bf16）
- 使用device_map="auto"进行自动设备分配

### 2. 速度优化
- 启用flash attention（如果支持）
- 使用多GPU并行处理
- 调整num_workers参数

### 3. 精度优化
- 使用fp32进行高精度测试
- 检查特征数值范围是否合理
- 验证projector输出维度是否正确

## 常见问题

### Q1: 如何确定projector输出维度？
A: 检查配置文件中的`hidden_size`参数，projector输出维度应该与语言模型的hidden_size匹配。

### Q2: 特征值范围异常怎么办？
A: 检查projector的初始化方式，可能需要调整权重初始化或添加归一化层。

### Q3: 如何比较不同projector的效果？
A: 使用相同的输入数据，比较不同projector输出的特征统计信息（均值、标准差、分布等）。

### Q4: 支持哪些vision tower？
A: EMOVA支持多种vision tower，包括CLIP、InternViT、UNIT等，具体参考配置文件示例。

## 示例脚本

运行示例脚本查看详细用法：
```bash
python EMOVA/scripts/example_projector_test.py
```

这个脚本会展示如何创建测试数据、运行推理和分析结果。

## 总结

通过使用`--projector_only`参数，你可以：

1. **快速测试**：只加载必要的组件，节省内存和时间
2. **调试训练**：专注于第一阶段的训练效果
3. **特征分析**：深入分析projector的输出特征
4. **性能评估**：评估不同projector架构的效果

这对于LLaVA第一阶段的训练和调试非常有用。 