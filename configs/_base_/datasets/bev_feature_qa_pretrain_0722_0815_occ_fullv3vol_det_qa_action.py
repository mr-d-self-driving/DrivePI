data_args = dict(
    data_path=[
        # 28k x 3
        "/path/DrivePI_Data/drivepi_captions/nuscenes_train_annotation_front_28130_0815.json",  # BEV feature pickle file
        "/path/DrivePI_Data/drivepi_captions/nuscenes_train_annotation_back_28130_0815.json",  # BEV feature pickle file
        "/path/DrivePI_Data/drivepi_captions/nuscenes_train_annotation_all_28130_0722.json",  # BEV feature pickle file

        # occ
        "/path/DrivePI_Data/drivepi_captions/occ_class_281300.json",  # BEV feature pickle file
        # "/path/DrivePI_Data/drivepi_captions/occ_class_1007_extra_140650.json",  # BEV feature pickle file
        "/path/DrivePI_Data/drivepi_captions/occflow_class_1008_140k.json",  # 140650
        "/path/DrivePI_Data/drivepi_captions/occ_yorn_140650.json",  # BEV feature pickle file

        "/path/DrivePI_Data/drivepi_captions/nuscenes_train_det_3d_new.json", # 184k

        #  eval
        "/path/DrivePI_Data/drivepi_captions/nusceneqa_train_llava.json", #

        #  eval
        "/path/DrivePI_Data/drivepi_captions/nuscenes_action_train.json", # 28k.
    ],
    bev_feature_folder="/path/DrivePI_Data/unilion_bev_feats_train/",
    lazy_preprocess=True,
    is_multimodal=False,
    image_aspect_ratio='square',
    feature_hidden_size=384,  # BEV feature dimension - matches model config
)