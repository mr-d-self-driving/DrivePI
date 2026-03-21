_base_ = [
    '../../_base_/models/qwen2_5_empty_vision.py',
    '../../_base_/datasets/bev_feature_qa_pretrain_0722_0815_occ_fullv3vol_det_qa_action.py',
    '../../_base_/training/default.py'
]

model_args = dict(
    version="qwen2",

    pretrain_mm_mlp_adapter="/path/work_dirs/emova_llava-qwen2_5-0.5b-bev-featurev2-sms-pretrain0722_three/mm_projector.bin",

    language_model=dict(
        type='EmovaQwen2ForCausalLM',
        pretrained_model_name_or_path='Qwen/Qwen2.5-0.5B-Instruct',
        attn_implementation="flash_attention_2",
        from_pretrained=True,
        hidden_fusion_method='none',  # Options: 'weighted_sum', 'gated', 'max_pooling', 'avg_pooling'
        learn_fusion_weights=True,  # Whether to learn fusion weights
        use_text_last_fetas=True,
        occ_loss_weight=1.0,
        occ_flow_loss_weight=1.0,
        trainable=True,
    ),
    mm_vision_tower=dict(trainable=False),
    mm_projector=dict(
        type='SelfMiningMMProjector',
        mlp_depth=2,
        downsample_rate=9,
        downsample_size=(60, 60),
        num_input_token=180 * 180,
        trainable=True,
    ),
    # Replace occ_head with bev_occ_estimator
    bev_occ_estimator=dict(
        type='OCCHead',
        hidden_channel=256,
        Dz=16,
        flow=True,  # Whether to predict flow field
        bev_occ_config=dict(
            input_scope=[[-51.2, 51.2, 0.4], [-51.2, 51.2, 0.4]],
            output_scope=[[-51.2, 51.2, 0.4], [-51.2, 51.2, 0.4]],
            prescale_factor=1.0
        ),
        use_mask=True,
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
            'manmade', 'vegetation', 'free'
        ],
        num_classes=17,
        size=[200, 200],
        class_balance=True,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0  # Use occ_loss_weight from config
        ),
        loss_occ_flow=dict(
            type='L1Loss',
            loss_weight=1.0  # Use occ_flow_loss_weight from config
        ),
        feature_proj=True,  # Project from LLM features to BEV features
        trainable=True,
        fusion_type='none'
    ),
    diff_anchor_planner_head=dict(
        type='DiffAnchorPlannerHead',
        planning_anchor='/path/data/nuscenes/kmeans_planning_6.npy',
        in_channels=sum([128, 128, 128]),
        hidden_channel=128,
        num_decoder_layers=1,
        planning_config=dict(
            grid_size=[360, 360, 32],
            out_size_factor=2
        ),
        loss_plan_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=5.0),
        loss_plan_reg=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        feature_proj=True,  # Project from LLM features to planning features
        trainable=True,
        fusion_type='none'
    ),
    occ_loss_weight=1.0,
    occ_flow_loss_weight=1.0,
    planning_loss_weight=1.0,
    enable_occ_prediction=True,
    enable_planning=True,  # Enable path planning prediction
)

data_args = dict(
    bev_feature_folder="/path/DrivePI_Data/unilion_bev_feats_train/",
    lazy_preprocess=True,
    is_multimodal=False,
    image_aspect_ratio='square',
    feature_hidden_size=384,  # BEV feature dimension - matches model config

    # Occupancy grid related configurations
    occ_grid_dir="/path/DrivePI_Data/saved_openocc_gt_occ_train",
    occ_flow_dir="/path/DrivePI_Data/saved_openocc_gt_occ_flow_train",
    include_occ_grid=True,
    include_occ_flow=True,

    # Path planning related configurations
    planning_dir="/path/DrivePI_Data/saved_action_train",
    include_planning=True,
)

training_args = dict(
    output_dir="/path/work_dirs/drivepi_05b_1e_4/",
    deepspeed="./scripts/zero3.json",
    bf16=True,
    fp16=False,
    tf32=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    save_on_each_node=True,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.4,
    lr_scheduler_type="cosine",
    logging_steps=10,
    model_max_length=4096+1024,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    group_by_modality_length=True,
    report_to="tensorboard"
)