#!/usr/bin/env python3
"""
示例脚本：如何使用修改后的推理脚本来测试LLaVA第一阶段的projector结果

这个脚本展示了两种使用方式：
1. 只测试projector输出（projector_only模式）
2. 完整的模型推理（默认模式）
"""

import json
import os
from PIL import Image
import numpy as np

def create_test_data():
    """创建测试数据"""
    # 创建一些测试图像（这里用随机数据模拟）
    test_images = []
    for i in range(3):
        # 创建一个简单的测试图像
        img = Image.new('RGB', (224, 224), color=(i*50, i*50, i*50))
        img_path = f"test_image_{i}.jpg"
        img.save(img_path)
        test_images.append(img_path)
    
    # 创建测试数据
    test_data = []
    for i, img_path in enumerate(test_images):
        test_data.append({
            "image": img_path,
            "question": f"Describe what you see in image {i}."
        })
    
    # 保存测试数据
    with open("test_data.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    return test_data

def test_projector_only():
    """测试只使用projector的模式"""
    print("=== 测试 Projector-Only 模式 ===")
    
    # 创建测试数据
    test_data = create_test_data()
    
    # 运行projector-only推理（进行完整QA）
    cmd1 = f"""python EMOVA/scripts/emova_batch_inference_projector.py \\
        --config your_config_file.py \\
        --json_path test_data.json \\
        --projector_output_path qa_results.json \\
        --projector_only \\
        --batch_size 2 \\
        --verbose"""
    
    # 运行projector-only推理（输出projector特征用于调试）
    cmd2 = f"""python EMOVA/scripts/emova_batch_inference_projector.py \\
        --config your_config_file.py \\
        --json_path test_data.json \\
        --projector_output_path projector_features.json \\
        --projector_only \\
        --batch_size 2 \\
        --verbose"""
    
    # 运行projector-only推理（指定权重路径）
    cmd3 = f"""python EMOVA/scripts/emova_batch_inference_projector.py \\
        --config your_config_file.py \\
        --json_path test_data.json \\
        --projector_output_path qa_results.json \\
        --projector_weights_path /path/to/your/mm_projector.bin \\
        --projector_only \\
        --batch_size 2 \\
        --verbose"""
    
    print("运行命令（进行完整QA推理）:")
    print(cmd1)
    print("\n运行命令（输出projector特征用于调试）:")
    print(cmd2)
    print("\n运行命令（指定权重路径进行QA推理）:")
    print(cmd3)
    print("\n这些命令会:")
    print("1. 按照训练时的初始化方法初始化完整模型")
    print("2. 加载训练好的projector权重")
    print("3. 根据输出文件名决定:")
    print("   - 如果文件名包含'projector'：输出projector特征用于调试")
    print("   - 否则：进行完整的QA推理并输出文本回复")
    
    # 预期的输出格式
    print(f"\nQA推理输出格式示例:")
    qa_output = [
        "The image shows a cat sitting on a windowsill.",
        "There are several cars parked in the parking lot."
    ]
    print(json.dumps(qa_output, indent=2))
    
    print(f"\nProjector特征输出格式示例（调试用）:")
    feature_output = {
        "projected_features": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
        "feature_shape": [256, 4096],
        "original_image_size": [224, 224]
    }
    print(json.dumps(feature_output, indent=2))

def test_full_model():
    """测试完整模型推理"""
    print("\n=== 测试完整模型推理模式 ===")
    
    # 运行完整模型推理
    cmd = f"""python EMOVA/scripts/emova_batch_inference_projector.py \\
        --config your_config_file.py \\
        --json_path test_data.json \\
        --output_path full_model_results.json \\
        --batch_size 2 \\
        --verbose"""
    
    print("运行命令:")
    print(cmd)
    print("\n这个命令会:")
    print("1. 加载完整的EMOVA模型（包括语言模型）")
    print("2. 处理图像和文本输入")
    print("3. 生成文本回复")
    print("4. 保存生成的回复到 full_model_results.json")

def analyze_projector_output():
    """分析projector输出的示例代码"""
    print("\n=== 分析 Projector 输出 ===")
    
    # 读取projector输出
    with open("projector_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    
    print("分析projector输出:")
    for i, result in enumerate(results):
        features = np.array(result['projected_features'])
        shape = result['feature_shape']
        img_size = result['original_image_size']
        
        print(f"样本 {i}:")
        print(f"  - 特征形状: {shape}")
        print(f"  - 原始图像尺寸: {img_size}")
        print(f"  - 特征统计: mean={features.mean():.4f}, std={features.std():.4f}")
        print(f"  - 特征范围: [{features.min():.4f}, {features.max():.4f}]")
        print()

def main():
    """主函数"""
    print("EMOVA Projector 测试示例")
    print("=" * 50)
    
    # 测试projector-only模式
    test_projector_only()
    
    # 测试完整模型模式
    test_full_model()
    
    # 分析输出示例
    analyze_projector_output()
    
    print("\n使用说明:")
    print("1. 将 'your_config_file.py' 替换为你的实际配置文件路径")
    print("2. 确保配置文件中的 mm_projector 设置正确")
    print("3. 根据需要调整 batch_size 和其他参数")
    print("4. projector_only 模式适合测试第一阶段的训练效果")
    print("5. 完整模式适合测试端到端的性能")

if __name__ == "__main__":
    main() 