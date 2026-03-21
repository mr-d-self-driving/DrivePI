model_args = dict(
    version="qwen2.5",  # Note that, in pretrain stage, version='plain'.
    freeze_backbone=False,

    pretrain_mm_mlp_adapter=None,
    mm_use_im_start_end=False,
    mm_use_im_patch_token=False,

    mm_patch_merge_type='flat',

    language_model=dict(
        type='EmovaQwen2ForCausalLM',
        pretrained_model_name_or_path='Qwen/Qwen2.5-3B-Instruct',
        attn_implementation="flash_attention_2",
        from_pretrained=True,
    ),
    mm_vision_tower=dict(
        type='EmptyVisionTower',
        hidden_size=384,  # Should match the feature dimension from dataset
        spatial_size=(180, 180),  # Spatial size for BEV features
    ),
    mm_projector=dict(
        type='ConcatChannelMMProjector',
        mlp_depth=2,
        downsample_rate=9,
        downsample_size=(60, 60),
        num_input_token=180 * 180,
    ),
)
