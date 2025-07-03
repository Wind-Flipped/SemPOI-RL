#!/usr/bin/env python3
"""
测试新的TravelTextDataset功能的脚本
"""

import sys
import os
sys.path.append('/home/liuyq/2025summer/SPOT-Trip/code')

from data import TravelTextDataset, create_travel_text_dataset
from LLMs import TravelStyleGRPOTrainer
import argparse

class Args:
    """简单的配置类"""
    def __init__(self):
        self.dataset_name = "Foursquare"  # 或 "Yelp"
        self.device = "cuda"

def test_travel_text_dataset():
    """测试TravelTextDataset的功能"""
    print("=== 测试TravelTextDataset ===")
    
    # 创建配置
    args = Args()
    
    try:
        # 创建文本数据集
        print("创建文本数据集...")
        text_dataset = create_travel_text_dataset(args, args.dataset_name)
        
        print(f"文本数据集大小: {len(text_dataset)}")
        
        # 显示前几个样本
        print("\n前3个文本样本:")
        for i in range(min(3, len(text_dataset))):
            item = text_dataset[i]
            print(f"\n样本 {i+1}:")
            print(f"用户ID: {item['uid']}")
            print(f"原始城市: {item['ori_region']}")
            print(f"目标城市: {item['dst_region']}")
            print(f"Hometown prompt (前200字符): {item['hometown_prompt'][:200]}...")
            print(f"Destination prompt (前200字符): {item['destination_prompt'][:200]}...")
        
        # 测试获取prompt和reference对
        print("\n=== 测试prompt-reference对生成 ===")
        prompts, references = text_dataset.get_prompt_reference_pairs()
        print(f"生成了 {len(prompts)} 个prompt-reference对")
        
        if len(prompts) > 0:
            print(f"\n第一个prompt样例 (前300字符):\n{prompts[0][:300]}...")
            print(f"\n第一个reference样例 (前300字符):\n{references[0][:300]}...")
        
        return text_dataset
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_grpo_trainer_with_text_dataset(text_dataset):
    """测试GRPO训练器与文本数据集的集成"""
    print("\n=== 测试GRPO训练器集成 ===")
    
    try:
        # 创建训练器
        print("初始化GRPO训练器...")
        trainer = TravelStyleGRPOTrainer(
            model_name="/home/liuyq/X-R1/Qwen3-8B",
            device="cuda",
            use_lora=True
        )
        
        # 测试prepare_dataset方法
        print("测试prepare_dataset方法...")
        dataset, references = trainer.prepare_dataset(text_dataset=text_dataset)
        
        print(f"准备的数据集大小: {len(dataset)}")
        print(f"参考答案数量: {len(references)}")
        
        if len(dataset) > 0:
            print(f"第一个数据集样例: {dataset[0]}")
        
        print("集成测试成功!")
        
    except Exception as e:
        print(f"集成测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("开始测试新的TravelTextDataset功能...")
    
    # 测试文本数据集
    text_dataset = test_travel_text_dataset()
    
    # 如果文本数据集创建成功，测试GRPO集成
    if text_dataset is not None and len(text_dataset) > 0:
        test_grpo_trainer_with_text_dataset(text_dataset)
    else:
        print("文本数据集创建失败，跳过GRPO集成测试")
    
    print("\n测试完成!")
