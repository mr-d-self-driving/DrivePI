#!/usr/bin/env python3
"""
Script to prepare BEV features and QA data for training.
"""

import os
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm
import argparse
import re


def clean_image_paths(image_paths_str):
    """Clean and split the concatenated image paths string."""
    paths = []
    
    camera_patterns = [
        'samples/CAM_FRONT/',
        'samples/CAM_FRONT_RIGHT/',
        'samples/CAM_FRONT_LEFT/',
        'samples/CAM_BACK/',
        'samples/CAM_BACK_LEFT/',
        'samples/CAM_BACK_RIGHT/'
    ]
    
    for pattern in camera_patterns:
        if pattern in image_paths_str:
            start_idx = image_paths_str.find(pattern)
            if start_idx != -1:
                end_idx = len(image_paths_str)
                for other_pattern in camera_patterns:
                    if other_pattern != pattern:
                        next_idx = image_paths_str.find(other_pattern, start_idx + 1)
                        if next_idx != -1 and next_idx < end_idx:
                            end_idx = next_idx
                
                path = image_paths_str[start_idx:end_idx].strip()
                if path and path not in paths:
                    paths.append(path)
    
    return paths


def process_qa_data(qa_file_path, output_file_path):
    """Process QA data and clean up the format."""
    with open(qa_file_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    processed_data = []
    
    for item in tqdm(qa_data, desc="Processing QA data"):
        processed_item = {}
        
        # Extract scene token
        if 'history' in item and 'scene_token' in item['history']:
            processed_item['token'] = item['history']['scene_token']
        elif 'token' in item:
            processed_item['token'] = item['token']
        else:
            continue
        
        # Clean image paths
        if 'image' in item:
            if isinstance(item['image'], list):
                processed_item['image_paths'] = item['image']
            elif isinstance(item['image'], str):
                processed_item['image_paths'] = clean_image_paths(item['image'])
        
        # Process conversations
        if 'conversations' in item:
            processed_item['conversations'] = item['conversations']
        else:
            prediction = item.get('predictions', '')
            processed_item['conversations'] = [
                {
                    'from': 'human',
                    'value': '<image>\nPlease describe what you see in this scene.'
                },
                {
                    'from': 'gpt',
                    'value': prediction if prediction else 'I can see the scene from the BEV perspective.'
                }
            ]
        
        # Add additional metadata
        if 'history' in item:
            processed_item['history'] = item['history']
        
        if 'predictions' in item:
            processed_item['predictions'] = item['predictions']
        
        processed_data.append(processed_item)
    
    # Save processed data
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(processed_data)} QA items")
    print(f"Saved to: {output_file_path}")
    
    return processed_data


def create_bev_features_from_qa(qa_file_path, feature_folder, feature_dim=384, spatial_size=(180, 180)):
    """Create BEV features based on QA data tokens."""
    with open(qa_file_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    os.makedirs(feature_folder, exist_ok=True)
    
    for item in tqdm(qa_data, desc="Creating BEV features"):
        token = None
        if 'history' in item and 'scene_token' in item['history']:
            token = item['history']['scene_token']
        elif 'token' in item:
            token = item['token']
        
        if not token:
            continue
        
        feature_path = os.path.join(feature_folder, f"{token}.pt")
        dummy_feature = torch.randn(num_patches, feature_dim)
        torch.save(dummy_feature, feature_path)
    
    print(f"Created BEV features in {feature_folder}")


def main():
    parser = argparse.ArgumentParser(description="Prepare BEV features and QA data for training")
    parser.add_argument("--qa_file", type=str, required=True,
                       help="Path to QA JSON file")
    parser.add_argument("--processed_qa_file", type=str, required=True,
                       help="Output path for processed QA file")
    parser.add_argument("--bev_feature_folder", type=str, required=True,
                       help="Folder to store BEV features")
    parser.add_argument("--feature_dim", type=int, default=1024,
                       help="BEV feature dimension")
    parser.add_argument("--num_patches", type=int, default=576,
                       help="Number of spatial patches in BEV feature")
    parser.add_argument("--create_dummy_features", action="store_true",
                       help="Create dummy BEV features for testing")
    
    args = parser.parse_args()
    
    # Process QA data
    print("Processing QA data...")
    processed_qa_data = process_qa_data(args.qa_file, args.processed_qa_file)
    
    # Create dummy features if requested
    if args.create_dummy_features:
        print("Creating dummy BEV features...")
        create_bev_features_from_qa(
            args.processed_qa_file,
            args.bev_feature_folder,
            feature_dim=args.feature_dim,
            num_patches=args.num_patches
        )
    
    # Display sample data
    print("\nSample processed QA data:")
    for i, item in enumerate(processed_qa_data[:2]):
        print(f"\nSample {i + 1}:")
        print(f"  Token: {item['token']}")
        if 'image_paths' in item:
            print(f"  Image paths: {len(item['image_paths'])} images")
        if 'conversations' in item:
            print("  Conversations:")
            for conv in item['conversations']:
                print(f"    {conv['from']}: {conv['value'][:100]}...")


if __name__ == "__main__":
    main() 