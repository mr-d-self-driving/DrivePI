#!/usr/bin/env python3
"""
Test script to verify the BEV feature system is working correctly.
"""

import os
import torch
import json
import argparse
from pathlib import Path


def test_empty_vision_encoder():
    """Test the EmptyVisionTower encoder."""
    print("Testing EmptyVisionTower...")
    
    try:
        from emova.model.multimodal_encoder.empty_encoder import EmptyVisionTower
        
        # Create encoder
        encoder = EmptyVisionTower(hidden_size=384, spatial_size=(180, 180))
        
        # Create dummy features
        batch_size = 2
        num_patches = 180 * 180
        feature_dim = 384
        
        dummy_features = torch.randn(batch_size, num_patches, feature_dim)
        
        # Test forward pass
        output = encoder(dummy_features)
        
        assert output.shape == dummy_features.shape, f"Shape mismatch: {output.shape} vs {dummy_features.shape}"
        assert torch.allclose(output, dummy_features), "Output should be identical to input"
        
        print("✓ EmptyVisionTower test passed")
        return True
        
    except Exception as e:
        print(f"✗ EmptyVisionTower test failed: {e}")
        return False


def test_concat_channel_projector():
    """Test the ConcatChannelMMProjector."""
    print("Testing ConcatChannelMMProjector...")
    
    try:
        from emova.model.multimodal_projector.builder import ConcatChannelMMProjector
        
        # Create projector
        projector = ConcatChannelMMProjector(
            mm_hidden_size=384,
            hidden_size=1024,
            mlp_depth=2,
            downsample_rate=3,
            downsample_size=(60, 60),
            num_input_token=180 * 180
        )
        
        # Create dummy features
        batch_size = 2
        num_patches = 180 * 180
        feature_dim = 384
        
        dummy_features = torch.randn(batch_size, num_patches, feature_dim)
        
        # Test forward pass
        output = projector(dummy_features)
        
        expected_output_patches = 60 * 60  # downsample_size
        assert output.shape[1] == expected_output_patches, f"Expected {expected_output_patches} patches, got {output.shape[1]}"
        assert output.shape[2] == 1024, f"Expected hidden_size 1024, got {output.shape[2]}"
        
        print("✓ ConcatChannelMMProjector test passed")
        return True
        
    except Exception as e:
        print(f"✗ ConcatChannelMMProjector test failed: {e}")
        return False


def test_feature_dataset():
    """Test the LazyFeatureDataset."""
    print("Testing LazyFeatureDataset...")
    
    try:
        from emova.data.feature_dataset import LazyFeatureDataset
        
        # Create dummy data
        temp_dir = Path("./temp_test_data")
        temp_dir.mkdir(exist_ok=True)
        
        # Create dummy features
        feature_dir = temp_dir / "bev_features"
        feature_dir.mkdir(exist_ok=True)
        
        num_patches = 180 * 180
        feature_dim = 384
        
        # Create a few dummy feature files
        for i in range(3):
            token = f"test_token_{i}"
            features = torch.randn(num_patches, feature_dim)
            torch.save(features, feature_dir / f"{token}.pt")
        
        # Create dummy QA data
        qa_data = [
            {
                "history": {"scene_token": "test_token_0"},
                "conversations": [
                    {"from": "human", "value": "Describe this scene."},
                    {"from": "gpt", "value": "This is a test scene."}
                ]
            },
            {
                "history": {"scene_token": "test_token_1"},
                "conversations": [
                    {"from": "human", "value": "What do you see?"},
                    {"from": "gpt", "value": "I see a test scene."}
                ]
            }
        ]
        
        qa_file = temp_dir / "test_qa.json"
        with open(qa_file, 'w') as f:
            json.dump(qa_data, f)
        
        # Create dataset
        dataset = LazyFeatureDataset(
            data_path=str(temp_dir / "dummy.pkl"),
            bev_feature_folder=str(feature_dir),
            qa_file=str(qa_file),
            feature_hidden_size=feature_dim
        )
        
        # Test dataset
        assert len(dataset) > 0, "Dataset should not be empty"
        
        # Test getting an item
        item = dataset[0]
        assert 'conversations' in item, "Item should have conversations"
        assert 'images' in item, "Item should have images"
        
        print("✓ LazyFeatureDataset test passed")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ LazyFeatureDataset test failed: {e}")
        return False


def test_model_integration():
    """Test the complete model integration."""
    print("Testing model integration...")
    
    try:
        from emova.model.multimodal_encoder.empty_encoder import EmptyVisionTower
        from emova.model.multimodal_projector.builder import ConcatChannelMMProjector
        
        # Create encoder and projector
        encoder = EmptyVisionTower(hidden_size=384, spatial_size=(180, 180))
        projector = ConcatChannelMMProjector(
            mm_hidden_size=384,
            hidden_size=1024,
            mlp_depth=2,
            downsample_rate=3,
            downsample_size=(60, 60),
            num_input_token=180 * 180
        )
        
        # Create dummy features
        batch_size = 2
        num_patches = 180 * 180
        feature_dim = 384
        
        dummy_features = torch.randn(batch_size, num_patches, feature_dim)
        
        # Test complete pipeline
        encoded_features = encoder(dummy_features)
        projected_features = projector(encoded_features)
        
        assert projected_features.shape == (batch_size, 3600, 1024), f"Unexpected output shape: {projected_features.shape}"
        
        print("✓ Model integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Model integration test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test BEV feature system")
    parser.add_argument("--test", type=str, choices=["all", "encoder", "projector", "dataset", "integration"],
                       default="all", help="Which test to run")
    
    args = parser.parse_args()
    
    print("BEV Feature System Test")
    print("=" * 50)
    
    tests = []
    
    if args.test in ["all", "encoder"]:
        tests.append(("EmptyVisionTower", test_empty_vision_encoder))
    
    if args.test in ["all", "projector"]:
        tests.append(("ConcatChannelMMProjector", test_concat_channel_projector))
    
    if args.test in ["all", "dataset"]:
        tests.append(("LazyFeatureDataset", test_feature_dataset))
    
    if args.test in ["all", "integration"]:
        tests.append(("Model Integration", test_model_integration))
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The BEV feature system is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 