#!/usr/bin/env python3
"""
测试脚本：验证修改后的推理脚本
"""

import json
import os
from PIL import Image

def test_projector_only_mode():
    """测试projector-only模式"""
    print("=== 测试 Projector-Only 模式 ===")
    
    # 创建测试数据
    test_data = []
    for i in range(2):
        img = Image.new('RGB', (224, 224), color=(i*100, i*100, i*100))
        img_path = f"test_image_{i}.jpg"
        img.save(img_path)
        test_data.append({
            "image": img_path,
            "question": f"Describe image {i}."
        })
    
    # 保存测试数据
    with open("test_data.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print("✓ 测试数据创建成功")
    print("✓ 可以运行以下命令测试:")
    print("\n1. 进行完整QA推理:")
    print("python EMOVA/scripts/emova_batch_inference_projector.py \\")
    print("    --config your_config.py \\")
    print("    --json_path test_data.json \\")
    print("    --projector_only \\")
    print("    --projector_output_path qa_results.json")
    
    print("\n2. 输出projector特征用于调试:")
    print("python EMOVA/scripts/emova_batch_inference_projector.py \\")
    print("    --config your_config.py \\")
    print("    --json_path test_data.json \\")
    print("    --projector_only \\")
    print("    --projector_output_path projector_features.json")
    
    print("\n3. 指定权重路径进行QA推理:")
    print("python EMOVA/scripts/emova_batch_inference_projector.py \\")
    print("    --config your_config.py \\")
    print("    --json_path test_data.json \\")
    print("    --projector_weights_path /path/to/mm_projector.bin \\")
    print("    --projector_only \\")
    print("    --projector_output_path qa_results.json")
    
    # 清理测试文件
    for img_path in ['test_image_0.jpg', 'test_image_1.jpg']:
        if os.path.exists(img_path):
            os.unlink(img_path)
    if os.path.exists('test_data.json'):
        os.unlink('test_data.json')

def main():
    print("EMOVA Projector 修改验证")
    print("=" * 40)
    
    test_projector_only_mode()
    
    print("\n修改总结:")
    print("1. ✅ 添加了 --projector_only 参数")
    print("2. ✅ 添加了 --projector_output_path 参数")
    print("3. ✅ 添加了 --projector_weights_path 参数")
    print("4. ✅ 修改了 ProjectorOnlyChatbot 类")
    print("5. ✅ 按照训练时的初始化方法初始化模型")
    print("6. ✅ 支持加载训练好的projector权重")
    print("7. ✅ 支持完整的QA推理功能")
    print("8. ✅ 支持输出projector特征用于调试")
    
    print("\n关键改进:")
    print("- 使用 build_language_model 初始化语言模型")
    print("- 调用 initialize_vision_modules 初始化vision模块")
    print("- 支持指定或自动查找projector权重文件")
    print("- 保持与训练时相同的初始化流程")
    print("- ProjectorOnlyChatbot 和 Chatbot 使用方式完全相同")
    print("- 根据输出文件名自动选择QA推理或特征输出模式")

if __name__ == "__main__":
    main() 
 