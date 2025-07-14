#!/usr/bin/env python3
"""
prepare_prompts.py - 数据集准备脚本

该脚本负责：
1. 加载原始旅游数据
2. 使用LLM批量生成参考答案
3. 创建并保存训练数据集
4. 支持多卡并行加速生成过程
"""

import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'  # 设置可见GPU设备

sys.path.append('/home/wangb/lyq/2025summer/SPOT-Trip/code')

import argparse
import time
from datetime import datetime
from data import create_travel_text_dataset
from LLMs import TravelStyleGRPOTrainer
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Args:
    """配置类"""

    def __init__(self, dataset_name="Foursquare", device="cuda"):
        self.dataset_name = dataset_name
        self.device = device


def main():
    """主函数 - 数据集准备"""
    parser = argparse.ArgumentParser(description='准备旅游风格强化学习数据集')

    # 添加命令行参数
    parser.add_argument('--dataset_name', type=str, default='Foursquare',
                        choices=['Foursquare', 'Yelp'],
                        help='数据集名称 (default: Foursquare)')
    parser.add_argument('--model_name', type=str, default='../LLMs/Qwen3-8B',
                        help='LLM模型路径 (default: ../LLMs/Qwen3-8B)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (default: cuda)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批处理大小 (default: 4)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大处理样本数，用于测试 (default: None，处理全部)')
    parser.add_argument('--save_dataset', action='store_true', default=True,
                        help='是否保存数据集 (default: True)')
    parser.add_argument('--output_name', type=str, default=None,
                        help='输出数据集名称 (default: 自动生成)')

    args = parser.parse_args()

    print(f"\n{'=' * 100}")
    print(f"开始准备旅游风格强化学习数据集")
    print(f"{'=' * 100}")
    print(f"数据集: {args.dataset_name}")
    print(f"LLM模型: {args.model_name}")
    print(f"批处理大小: {args.batch_size}")
    print(f"最大样本数: {args.max_samples if args.max_samples else '全部'}")
    print(f"保存数据集: {args.save_dataset}")
    print(f"{'=' * 100}\n")

    # 记录开始时间
    start_time = time.time()

    try:
        # 1. 加载原始数据集
        print("🔄 加载原始数据集...")
        config = Args(args.dataset_name, args.device)
        text_dataset = create_travel_text_dataset(config, args.dataset_name)

        print(f"✅ 原始数据集加载成功!")
        print(f"   数据集大小: {len(text_dataset)}")
        print(f"   数据集类型: {type(text_dataset)}")

        # 限制样本数量（用于测试）
        if args.max_samples and args.max_samples < len(text_dataset):
            print(f"🔄 限制样本数量到 {args.max_samples} 个（测试模式）")

            # 创建子数据集
            class SubDataset:
                def __init__(self, original_dataset, max_samples):
                    self.original_dataset = original_dataset
                    self.max_samples = max_samples

                def __len__(self):
                    return min(self.max_samples, len(self.original_dataset))

                def __getitem__(self, idx):
                    if idx >= self.max_samples:
                        raise IndexError("Index out of range")
                    return self.original_dataset[idx]

            text_dataset = SubDataset(text_dataset, args.max_samples)
            print(f"   限制后数据集大小: {len(text_dataset)}")

        # 2. 初始化GRPO训练器（用于数据集准备）
        print("🔄 初始化数据处理器...")
        trainer = TravelStyleGRPOTrainer(
            model_name=args.model_name,
            device=args.device,
            use_lora=False  # 数据准备阶段不需要LoRA
        )
        print("✅ 数据处理器初始化完成!")

        # 3. 准备数据集（批量生成）
        print("🔄 开始批量生成训练数据集...")
        dataset, reference_responses = trainer.prepare_dataset(
            text_dataset=text_dataset,
            save_dataset=args.save_dataset,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size
        )

        # 4. 输出统计信息
        end_time = time.time()
        processing_time = end_time - start_time

        print(f"\n{'=' * 100}")
        print(f"数据集准备完成!")
        print(f"{'=' * 100}")
        print(f"✅ 处理统计:")
        print(f"   原始样本数: {len(text_dataset)}")
        print(f"   生成数据集大小: {len(dataset)}")
        print(f"   参考答案数量: {len(reference_responses)}")
        print(f"   处理时间: {processing_time:.2f} 秒 ({processing_time / 60:.1f} 分钟)")
        print(f"   平均处理速度: {len(dataset) / processing_time:.2f} 样本/秒")
        print(f"   批处理大小: {args.batch_size}")

        # 5. 显示数据集样例
        if len(dataset) > 0:
            print(f"\n📋 数据集样例:")
            for i in range(min(2, len(dataset))):
                print(f"\n样例 {i + 1}:")
                prompt = dataset[i]["prompt"]
                reference = dataset[i]["reference"]
                print(f"   Prompt (前200字符): {prompt[:200]}...")
                print(f"   Reference (前200字符): {reference[:200]}...")
                print(f"   Prompt长度: {len(prompt)} 字符")
                print(f"   Reference长度: {len(reference)} 字符")

        # 6. 保存位置信息
        if args.save_dataset:
            dataset_dir = "../dataset"
            print(f"\n💾 数据集保存信息:")
            print(f"   保存目录: {dataset_dir}")
            print(f"   可以在训练脚本中使用以下代码加载:")
            print(f"   ```python")
            print(f"   from datasets import load_from_disk")
            print(f"   dataset = load_from_disk('最新的数据集路径')")
            print(f"   ```")

        print(f"\n{'=' * 100}")
        print(f"数据集准备任务成功完成!")
        print(f"{'=' * 100}\n")

        return dataset, reference_responses

    except Exception as e:
        print(f"\n❌ 数据集准备失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    finally:
        # 清理GPU内存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 GPU内存已清理")
        except:
            pass


def prepare_small_test_dataset():
    """准备小规模测试数据集"""
    print("🔄 准备小规模测试数据集（10个样本）...")

    config = Args("Foursquare", "cuda")
    text_dataset = create_travel_text_dataset(config, "Foursquare")

    # 创建小规模测试数据集
    class SmallTestDataset:
        def __init__(self, original_dataset, size=10):
            self.original_dataset = original_dataset
            self.size = min(size, len(original_dataset))

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return self.original_dataset[idx]

    small_dataset = SmallTestDataset(text_dataset, 10)

    trainer = TravelStyleGRPOTrainer(
        model_name="../LLMs/Qwen3-8B",
        device="cuda",
        use_lora=False
    )

    dataset, references = trainer.prepare_dataset(
        text_dataset=small_dataset,
        save_dataset=True,
        batch_size=2
    )

    print(f"✅ 小规模测试数据集准备完成!")
    print(f"   数据集大小: {len(dataset)}")

    return dataset, references


if __name__ == "__main__":
    # 检查是否在交互模式下运行测试
    if len(sys.argv) == 1:
        print("提示：可以使用以下命令运行:")
        print("python prepare_prompts.py --help  # 查看所有选项")
        print("python prepare_prompts.py --max_samples 50 --batch_size 2  # 测试模式")
        print("python prepare_prompts.py --dataset_name Foursquare --batch_size 4  # 完整模式")
        print()

        # 如果没有参数，运行小规模测试
        user_input = input("是否运行小规模测试（10个样本）？[y/N]: ")
        if user_input.lower() in ['y', 'yes']:
            prepare_small_test_dataset()
        else:
            print("已取消运行")
    else:
        # 正常运行主函数
        main()
