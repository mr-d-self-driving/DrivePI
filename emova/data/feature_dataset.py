import json
import os
import random
import pickle
from dataclasses import dataclass
from typing import Dict, Sequence

import transformers
import torch
from torch.utils.data import Dataset

from emova.utils import rank0_print

from .preprocess import *
from ..constants import IGNORE_INDEX
import numpy as np


def read_data_file(file):
    if file.endswith('.json'):
        list_data_dict = json.load(open(file, "r"))
    elif file.endswith('.jsonl'):
        list_data_dict = [json.loads(line) for line in open(file, 'r', encoding='utf-8')]
    elif file.endswith('.pkl'):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        list_data_dict = data['infos'] if 'infos' in data else data
    else:
        raise RuntimeError(f"Unrecoginized file: {file}")
    return list_data_dict


class LazyFeatureDataset(Dataset):
    """Dataset for supervised fine-tuning with pre-computed BEV features and QA data."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(LazyFeatureDataset, self).__init__()

        if isinstance(data_path, list):
            rank0_print("Load annotations from list. {}".format(data_path))
            list_data_dict = []
            for sub_data_path in data_path:
                if os.path.isdir(sub_data_path):
                    for file in os.listdir(sub_data_path):
                        list_data_dict += read_data_file(os.path.join(sub_data_path, file))
                else:
                    list_data_dict += read_data_file(sub_data_path)
        else:
            list_data_dict = read_data_file(data_path)

        rank0_print(f"Dataset length: {len(list_data_dict)}.")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        
        # Set occupancy grid and flow data paths
        self.occ_grid_dir = getattr(data_args, 'occ_grid_dir', None)
        self.occ_flow_dir = getattr(data_args, 'occ_flow_dir', None)
        
        # Set planning data path
        self.planning_dir = getattr(data_args, 'planning_dir', None)
        
        # Check and print occupancy grid and flow data paths
        if self.occ_grid_dir:
            rank0_print(f"Using occupancy grid data from: {self.occ_grid_dir}")
        if self.occ_flow_dir:
            rank0_print(f"Using occupancy flow data from: {self.occ_flow_dir}")
        
        # Check and print planning data path
        if self.planning_dir:
            rank0_print(f"Using planning data from: {self.planning_dir}")
        
        # Get metadata path
        self.meta_dir = getattr(data_args, 'meta_dir', None)
        if self.meta_dir:
            rank0_print(f"Using meta data from: {self.meta_dir}")

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int) or isinstance(i, torch.Tensor) and i.numel() == 1:
            sources = [sources]

        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        gt_occ = None
        gt_occ_flow = None
        gt_planning = None
        meta_data = None
        
        if 'image' in sources[0]:
            token = sources[0]['history']['token']
            feature_folder = self.data_args.bev_feature_folder
            feature_path = os.path.join(feature_folder, f"{token}.npz")
            
            with np.load(feature_path) as data_bev:
                features = data_bev['data']

            image = torch.from_numpy(features[0])
            
            # Set dummy image size for compatibility
            image_size = (180, 180)  # Default size, can be overridden
            
            # Load occupancy grid data (if exists)
            if self.occ_grid_dir:
                occ_grid_path = os.path.join(self.occ_grid_dir, f"{token}.npz")
                if os.path.exists(occ_grid_path):
                    try:
                        with np.load(occ_grid_path) as data_occ:
                            gt_occ_data = data_occ['data']
                        gt_occ = torch.from_numpy(gt_occ_data)
                    except Exception as e:
                        rank0_print(f"Error loading occupancy grid for {token}: {e}")
                        gt_occ = None
            
            # Load occupancy flow data (if exists)
            if self.occ_flow_dir:
                occ_flow_path = os.path.join(self.occ_flow_dir, f"{token}.npz")
                if os.path.exists(occ_flow_path):
                    try:
                        with np.load(occ_flow_path) as data_occ_flow:
                            gt_occ_flow_data = data_occ_flow['data']
                        gt_occ_flow = torch.from_numpy(gt_occ_flow_data)
                    except Exception as e:
                        rank0_print(f"Error loading occupancy flow for {token}: {e}")
                        gt_occ_flow = None
                        
            # Load planning data (if exists)
            if self.planning_dir:
                planning_path = os.path.join(self.planning_dir, f"{token}.npy")
                if os.path.exists(planning_path):
                    try:
                        gt_plan_data = np.load(planning_path, allow_pickle=True).item()
                        if 'fut_boxes' in gt_plan_data.keys():
                            gt_plan_data.pop('fut_boxes')
                        gt_planning = {}
                        for plan_key in gt_plan_data.keys():
                            gt_planning[plan_key] = torch.from_numpy(gt_plan_data[plan_key])
                    except Exception as e:
                        rank0_print(f"Error loading planning data for {token}: {e}")
                        gt_planning = None
            
            # Load metadata (if exists)
            if self.meta_dir:
                meta_path = os.path.join(self.meta_dir, f"{token}.pkl")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'rb') as f:
                            meta_data = pickle.load(f)
                    except Exception as e:
                        rank0_print(f"Error loading meta data for {token}: {e}")
                        meta_data = None
            
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))

        if isinstance(i, int) or isinstance(i, torch.Tensor) and i.numel() == 1:
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image/bev_feature exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['image_size'] = image_size
            
            # Add occupancy grid data to return dict
            if gt_occ is not None:
                data_dict['gt_occ'] = gt_occ
            
            # Add occupancy flow data to return dict
            if gt_occ_flow is not None:
                data_dict['gt_occ_flow'] = gt_occ_flow
                
            # Add planning data to return dict
            if gt_planning is not None:
                data_dict['gt_planning'] = gt_planning
            
            # Add metadata to return dict
            if meta_data is not None:
                data_dict['meta_data'] = meta_data
                
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['image_size'] = (crop_size['height'], crop_size['width'])
        data_dict['token_name'] = token
        return data_dict


@dataclass
class DataCollatorForFeatureDataset(object):
    """Collate examples for supervised fine-tuning with features."""

    tokenizer: transformers.PreTrainedTokenizer
    image_aspect_ratio: str

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            image_sizes = [instance['image_size'] for instance in instances]

            # Handle feature tensors
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
            batch['image_sizes'] = image_sizes
            
            # Process occupancy grid data
            if 'gt_occ' in instances[0] and instances[0]['gt_occ'] is not None:
                gt_occs = [instance.get('gt_occ', None) for instance in instances]
                if all(x is not None and x.shape == gt_occs[0].shape for x in gt_occs):
                    batch['gt_occ'] = torch.stack(gt_occs)
                else:
                    # Filter out None values
                    valid_gt_occs = [x for x in gt_occs if x is not None]
                    if valid_gt_occs:
                        batch['gt_occ'] = valid_gt_occs
            
            # Process occupancy flow data
            if 'gt_occ_flow' in instances[0] and instances[0]['gt_occ_flow'] is not None:
                gt_occ_flows = [instance.get('gt_occ_flow', None) for instance in instances]
                if all(x is not None and x.shape == gt_occ_flows[0].shape for x in gt_occ_flows):
                    batch['gt_occ_flow'] = torch.stack(gt_occ_flows)
                else:
                    # Filter out None values
                    valid_gt_occ_flows = [x for x in gt_occ_flows if x is not None]
                    if valid_gt_occ_flows:
                        batch['gt_occ_flow'] = valid_gt_occ_flows
            
            # Process planning data
            if 'gt_planning' in instances[0] and instances[0]['gt_planning'] is not None:
                gt_plannings = [instance.get('gt_planning', None) for instance in instances]
                # Planning data might be dictionary or other complex structure, can't simply stack
                # Filter out None values
                valid_gt_plannings = [x for x in gt_plannings if x is not None]
                if valid_gt_plannings:
                    batch['gt_planning'] = valid_gt_plannings
                    
            # Process metadata
            if 'meta_data' in instances[0] and instances[0]['meta_data'] is not None:
                meta_datas = [instance.get('meta_data', None) for instance in instances]
                # Filter out None values
                valid_meta_datas = [x for x in meta_datas if x is not None]
                if valid_meta_datas:
                    batch['metas'] = valid_meta_datas

        # Add token_name to batch
        if 'token_name' in instances[0]:
            batch['token_name'] = [instance['token_name'] for instance in instances]
        return batch


def make_feature_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                        data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning with features."""
    # Set occupancy grid and flow data paths
    data_args.occ_grid_dir = getattr(data_args, 'occ_grid_dir', "/path/DrivePI_Data/saved_openocc_gt_occ_train")
    data_args.occ_flow_dir = getattr(data_args, 'occ_flow_dir', "/path/DrivePI_Data/saved_openocc_gt_occ_flow_train")
    
    # Set planning data path
    data_args.planning_dir = getattr(data_args, 'planning_dir', "/path/DrivePI_Data/saved_planning_train")
    
    # Check if paths exist
    if not os.path.exists(data_args.occ_grid_dir):
        rank0_print(f"Warning: Occupancy grid directory {data_args.occ_grid_dir} does not exist.")
    if not os.path.exists(data_args.occ_flow_dir):
        rank0_print(f"Warning: Occupancy flow directory {data_args.occ_flow_dir} does not exist.")
    if not os.path.exists(data_args.planning_dir):
        rank0_print(f"Warning: Planning directory {data_args.planning_dir} does not exist.")
    
    train_dataset = LazyFeatureDataset(tokenizer=tokenizer,
                                       data_path=data_args.data_path,
                                       data_args=data_args)

    SMOKE_TEST = bool(os.environ.get("SMOKE_TEST", 0))
    if SMOKE_TEST:
        dataset_len = 64
        train_dataset.list_data_dict = train_dataset.list_data_dict[:dataset_len]

    data_collator = DataCollatorForFeatureDataset(tokenizer=tokenizer,
                                                  image_aspect_ratio=data_args.image_aspect_ratio)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)