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