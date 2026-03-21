_base_ = [
    '../../_base_/models/qwen2_5_empty_vision.py',
    '../../_base_/datasets/bev_feature_qa_pretrain_0722_three.py',
    '../../_base_/training/default.py'
]

model_args = dict(
    version="plain",
    # language_model=dict(trainable=False),
    language_model=dict(
        type='EmovaQwen2ForCausalLM',
        pretrained_model_name_or_path='Qwen/Qwen2.5-0.5B-Instruct',
        attn_implementation="flash_attention_2",
        from_pretrained=True,
        trainable=False
    ),
    mm_vision_tower=dict(trainable=False),
    mm_projector = dict(
        type='SelfMiningMMProjector',
        mlp_depth=2,
        downsample_rate=9,
        downsample_size=(60, 60),
        num_input_token=180 * 180,
        trainable=True,
    ),
)


data_args = dict(
    data_path=[
        "/path/DrivePI_Data/drivepi_captions/nuscenes_train_annotation_front_28130_0722.json",  # BEV feature pickle file
        "/path/DrivePI_Data/drivepi_captions/nuscenes_train_annotation_back_28130_0722.json",  # BEV feature pickle file
        "/path/DrivePI_Data/drivepi_captions/nuscenes_train_annotation_all_28130_0722.json",  # BEV feature pickle file
    ],
    bev_feature_folder="/path/DrivePI_Data/unilion_bev_feats_train/",
    lazy_preprocess=True,
    is_multimodal=False,
    image_aspect_ratio='square',
    feature_hidden_size=384,  # BEV feature dimension - matches model config
)


training_args = dict(
    output_dir="/path/work_dirs/emova_llava-qwen2_5-0.5b-bev-featurev2-sms-pretrain0722_three/",
    deepspeed="./scripts/zero3.json",
    bf16=True,
    tf32=True,
    fp16=False,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=32,
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=24000,
    save_total_limit=1,
    learning_rate=1e-3,
    weight_decay=0.,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=1,
    model_max_length=4096+1024,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    report_to="tensorboard",
)
