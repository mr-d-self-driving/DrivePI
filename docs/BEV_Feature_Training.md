# BEV Feature Training Guide

This guide explains how to use the BEV (Bird's Eye View) feature dataset with QA data for training with EMOVA.

## Overview

The BEV feature dataset allows you to train models using pre-computed BEV features instead of raw images, combined with QA (Question-Answer) annotations. This is particularly useful for autonomous driving applications where you have BEV representations of scenes with corresponding textual descriptions.

## Architecture

The system uses:
- **EmptyVisionTower**: A pass-through encoder that directly returns BEV features
- **ConcatChannelMMProjector**: A projector that handles spatial downsampling of BEV features
- **QA Dataset**: Combines BEV features with question-answer pairs

## Data Format

### 1. BEV Feature Files

BEV features should be stored as PyTorch tensors (`.pt` files) with the following format:
- Shape: `[num_patches, feature_dim]`
- Example: `[32400, 384]` for 180x180 spatial layout with 384-dimensional features
- Spatial layout: Features are flattened from 2D spatial grid (H, W) to 1D sequence

### 2. QA Data Format

The QA data should be in JSON format with the following structure:

```json
[
  {
    "history": {
      "scene_token": "unique_scene_token",
      "token": "sample_token",
      "camera_token": ["cam1", "cam2", ...]
    },
    "image": "concatenated_image_paths_string",
    "predictions": "description of the scene",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\n<image>\n<image>\n<image>\n<image>\n<image>\nPlease describe this scene."
      },
      {
        "from": "gpt",
        "value": "A building with a large glass window. The building has a sign that says 'protect what matters'. There are people standing in front of the building."
      }
    ]
  }
]
```

## Configuration

### 1. Model Configuration

The model uses an empty vision encoder with ConcatChannelMMProjector:

```python
# configs/_base_/models/qwen2_5_empty_vision.py
model_args = dict(
    version="qwen2.5",
    language_model=dict(
        type='EmovaQwen2ForCausalLM',
        pretrained_model_name_or_path='Qwen/Qwen2.5-3B-Instruct',
        from_pretrained=True,
    ),
    mm_vision_tower=dict(
        type='EmptyVisionTower',
        hidden_size=384,  # Feature dimension
        spatial_size=(180, 180),  # Spatial layout
    ),
    mm_projector=dict(
        type='ConcatChannelMMProjector',
        mlp_depth=2,
        downsample_rate=3,
        downsample_size=(60, 60),
        num_input_token=180 * 180,
    ),
)
```

### 2. Dataset Configuration

```python
# configs/_base_/datasets/bev_feature_qa_pretrain.py
data_args = dict(
    data_path="/data/nuscenes/nuscenes_infos_train.pkl",
    bev_feature_folder="/data/nuscenes/bev_features/",
    qa_file="/data/runhui/captions/nuscenes_train_28130_caption.json",
    lazy_preprocess=True,
    is_multimodal=False,
    image_aspect_ratio='square',
    feature_hidden_size=384,
)
```

## Data Preparation

### 1. Process QA Data

Use the provided script to process your QA data:

```bash
python scripts/prepare_bev_qa_data.py \
    --qa_file /data/runhui/captions/nuscenes_train_28130_caption.json \
    --processed_qa_file /data/runhui/captions/processed_qa_data.json \
    --bev_feature_folder /data/nuscenes/bev_features/ \
    --feature_dim 384 \
    --spatial_size 180 180 \
    --create_dummy_features
```

### 2. Complete Pipeline

For a complete pipeline from data preparation to training:

```bash
python scripts/run_bev_qa_training.py \
    --qa_file /data/runhui/captions/nuscenes_train_28130_caption.json \
    --processed_qa_file /data/runhui/captions/processed_qa_data.json \
    --bev_feature_folder /data/nuscenes/bev_features/ \
    --feature_dim 384 \
    --spatial_size 180 180
```

## Training

### 1. Start Training

Use the provided configuration to start training:

```bash
python -m emova.train.train \
    --config configs/example/llava/4dllm/pretrained.py
```

### 2. Training Configuration

The training configuration includes:

```python
# configs/example/llava/4dllm/pretrained.py
_base_ = [
    '../../../_base_/models/qwen2_5_empty_vision.py',
    '../../../_base_/datasets/bev_feature_qa_pretrain.py',
    '../../../_base_/training/default.py'
]

model_args = dict(
    version="plain",
    language_model=dict(trainable=False),
    mm_vision_tower=dict(trainable=False),  # Empty encoder doesn't need training
    mm_projector=dict(trainable=True)
)

training_args = dict(
    output_dir="./logdir/emova_llava-qwen2_5-3b-bev-feature-pretrain/",
    # ... other training parameters
)
```

## Implementation Details

### 1. Empty Vision Encoder

The `EmptyVisionTower` class is designed to work with spatial BEV features:

```python
@VISION_TOWER.register_module()
class EmptyVisionTower(BaseModalityEncoder):
    def __init__(self, hidden_size=384, spatial_size=(180, 180), **kwargs):
        super().__init__()
        self._hidden_size = hidden_size
        self._spatial_size = spatial_size
        
    def forward(self, features):
        # Ensure features have correct spatial layout
        # Shape: (batch_size, H*W, feature_dim)
        return features
```

### 2. ConcatChannelMMProjector

The projector handles spatial downsampling:

- **Input**: `(batch_size, 32400, 384)` - 180x180 spatial features
- **Downsample**: 3x3 pooling to `(batch_size, 3600, 3456)` - 60x60 spatial features
- **Output**: `(batch_size, 3600, hidden_size)` - Final projected features

### 3. QA Dataset Integration

The dataset automatically:
- Loads BEV features based on scene tokens
- Merges QA conversations with BEV features
- Handles both nuScenes format and custom QA data

## Data Structure

Your data should be organized as follows:

```
/data/
├── nuscenes/
│   ├── nuscenes_infos_train.pkl          # nuScenes metadata
│   └── bev_features/                     # BEV feature files
│       ├── token1.pt                     # 32400 x 384 features
│       ├── token2.pt
│       └── ...
└── runhui/
    └── captions/
        ├── nuscenes_train_28130_caption.json  # Original QA data
        └── processed_qa_data.json             # Processed QA data
```

## Customization

### 1. Feature Dimensions

To change feature dimensions, update both model and dataset configurations:

```python
# Model config
mm_vision_tower=dict(
    type='EmptyVisionTower',
    hidden_size=512,  # New feature dimension
    spatial_size=(200, 200),  # New spatial size
)

# Dataset config
feature_hidden_size=512,  # Match model config
```

### 2. Spatial Layout

The spatial layout is defined by the `spatial_size` parameter:
- `(180, 180)`: 32400 patches
- `(200, 200)`: 40000 patches
- `(160, 160)`: 25600 patches

### 3. Downsampling

Adjust the projector's downsampling parameters:

```python
mm_projector=dict(
    type='ConcatChannelMMProjector',
    downsample_rate=4,  # 4x4 downsampling
    downsample_size=(45, 45),  # 180/4 = 45
    num_input_token=180 * 180,
)
```

## Example Usage

Here's a complete example:

1. **Prepare your data structure**:
```bash
mkdir -p /data/nuscenes/bev_features/
mkdir -p /data/runhui/captions/
```

2. **Process QA data and create features**:
```bash
python scripts/prepare_bev_qa_data.py \
    --qa_file /data/runhui/captions/nuscenes_train_28130_caption.json \
    --processed_qa_file /data/runhui/captions/processed_qa_data.json \
    --bev_feature_folder /data/nuscenes/bev_features/ \
    --feature_dim 384 \
    --spatial_size 180 180 \
    --create_dummy_features
```

3. **Start training**:
```bash
python -m emova.train.train \
    --config configs/example/llava/4dllm/pretrain.py
```

The system will automatically:
- Load BEV features based on scene tokens
- Merge QA conversations with the features
- Use the empty vision encoder to pass features through
- Apply spatial downsampling with the ConcatChannelMMProjector
- Train the language model with the projected features 