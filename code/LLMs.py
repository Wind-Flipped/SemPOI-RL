"""
LLMs.py - 大语言模型调用接口和强化学习训练模块

该模块包含：
1. 调用大语言模型生成旅游风格的接口
2. 使用GRPO方法对LLM进行强化学习的训练接口
3. 文本相似度计算和奖励函数
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'  # 设置可见GPU设备

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments
)
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import logging
from dataclasses import dataclass
from tqdm import tqdm
import time
from vllm import LLM, SamplingParams
import logging

# 导入accelerate库进行分布式训练
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

# 导入SwanLab用于实验记录
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    logging.warning("SwanLab not available. Install with: pip install swanlab")

# 导入prompt模块
from prompt import TravelTrajectory, TravelPromptFormatter

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TravelStyleGenerator:
    """旅游风格生成器 - 封装LLM调用"""
    
    def __init__(self, model_name: str = "../LLMs/Qwen3-8B", device: str = "cuda", use_vllm= False):
        """
        初始化旅游风格生成器

        Args:
            model_name: 预训练模型名称
            device: 计算设备
        """
        self.device = device
        self.model_name = model_name
        self.use_vllm = use_vllm
        # 禁用所有日志输出
        logging.getLogger("vllm").setLevel(logging.CRITICAL)

        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if use_vllm:
            self.sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=512)
            self.model = LLM(model=model_name, max_model_len=4096, tensor_parallel_size=2)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )

        logger.info(f"Travel style generator initialized with {model_name}")

    def get_output(self, messages: List[str], max_length: int = 256,
                   temperature: float = 0.7, batch_size: int = 4) -> List[str]:
        """
        批量生成文本输出

        Args:
            messages: 输入文本列表
            max_length: 最大生成长度
            temperature: 生成温度
            batch_size: 批处理大小

        Returns:
            生成的文本列表
        """
        results = []

        # 分批处理
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i:i + batch_size]
            batch_results = self._generate_batch_from_messages(batch_messages, max_length, temperature)
            results.extend(batch_results)

        return results

    def _generate_batch_from_messages(self, messages: List[str], max_length: int, temperature: float) -> List[str]:
        """
        内部批处理生成函数，处理原始消息文本

        Args:
            messages: 消息文本列表
            max_length: 最大生成长度
            temperature: 生成温度

        Returns:
            生成结果列表
        """
        try:
            # 构建所有chat格式的消息
            all_texts = []

            for message in messages:
                chat_messages = [{"role": "user", "content": message}]
                text = self.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                all_texts.append(text)
            if self.use_vllm:
                # 使用vLLM进行批量生成
                outputs = self.model.generate(
                    all_texts,
                    sampling_params=self.sampling_params
                )
                generated_texts = [output.outputs[0].text for output in outputs]
                return generated_texts

            # 批量编码（利用padding支持不同长度的输入）
            inputs = self.tokenizer(
                all_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            ).to(self.model.device)

            input_lengths = inputs['attention_mask'].sum(dim=1)  # 获取每个输入的实际长度（不包括padding）

            # 批量生成（模型自动利用多卡并行）
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=0.8,
                    top_k=20,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )

            # 创建掩码来提取生成部分

            outputs_only_generated = []
            for i, output in enumerate(outputs):
                input_len = input_lengths[i]
                output = outputs[i]
                outputs_only_generated.append(output[input_len:])

            # 批量解码（自动处理不同长度）
            generated_texts = self.tokenizer.batch_decode(
                outputs_only_generated,  # 直接传入列表
                skip_special_tokens=True
            )
            return generated_texts

        except Exception as e:
            print(f"Error in batch generation from messages: {e}")
            return messages  # 返回原始消息作为降级处理

class TravelStyleRewardCalculator:
    """旅游风格相似度计算和奖励函数"""

    def __init__(self, similarity_model: str = "../LLMs/Qwen3-Embedding-4B"):
        """
        初始化奖励计算器

        Args:
            similarity_model: 用于计算文本相似度的模型
        """
        self.similarity_model = SentenceTransformer(similarity_model, device="cuda:2")
        logger.info(f"Reward calculator initialized with {similarity_model}")

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度

        Args:
            text1: 第一个文本（预测的旅游风格）
            text2: 第二个文本（实际的旅游风格）

        Returns:
            相似度分数 (0-1)
        """
        try:
            # 编码文本
            embeddings = self.similarity_model.encode([text1, text2])

            # 计算余弦相似度
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

            # 确保相似度在0-1范围内
            similarity = max(0.0, min(1.0, similarity))

            return float(similarity)

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def calculate_reward(self, predicted_style: str, actual_style: str,
                        similarity_weight: float = 1.0, length_penalty: float = 0.1) -> float:
        """
        计算强化学习奖励

        Args:
            predicted_style: 预测的旅游风格
            actual_style: 实际的旅游风格
            similarity_weight: 相似度权重
            length_penalty: 长度惩罚系数

        Returns:
            奖励分数
        """
        # 基础相似度奖励
        similarity = self.calculate_similarity(predicted_style, actual_style)
        similarity_reward = similarity * similarity_weight

        # 长度惩罚（避免生成过长或过短的文本）
        predicted_length = len(predicted_style.split())
        actual_length = len(actual_style.split())
        length_diff = abs(predicted_length - actual_length) / max(actual_length, 1)
        length_penalty_score = max(0, 1 - length_diff) * length_penalty

        # 总奖励
        total_reward = similarity_reward + length_penalty_score

        logger.debug(f"Similarity: {similarity:.3f}, Length penalty: {length_penalty_score:.3f}, Total reward: {total_reward:.3f}")

        return total_reward

    def get_embedding(self, texts: List[str], embedding_dim: Optional[int] = None) -> np.ndarray:
        """
        批量获取文本的embedding

        Args:
            texts: 输入文本列表
            embedding_dim: 嵌入维度，如果指定则对embedding进行截断

        Returns:
            文本embedding的numpy数组，形状为(batch_size, embedding_dim)
        """
        try:
            # 使用similarity_model编码文本
            embeddings = self.similarity_model.encode(texts)

            # 如果指定了embedding_dim，则截断到指定维度
            if embedding_dim is not None:
                if embedding_dim > embeddings.shape[1]:
                    logger.warning(f"Requested embedding_dim {embedding_dim} is larger than model output {embeddings.shape[1]}")
                    # 如果请求的维度大于模型输出，则用零填充
                    padded_embeddings = np.zeros((embeddings.shape[0], embedding_dim))
                    padded_embeddings[:, :embeddings.shape[1]] = embeddings
                    embeddings = padded_embeddings
                else:
                    # 截断到指定维度
                    embeddings = embeddings[:, :embedding_dim]

            logger.debug(f"Generated embeddings for {len(texts)} texts with shape {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # 返回零embedding作为fallback
            fallback_dim = embedding_dim if embedding_dim is not None else 768  # 默认维度
            return np.zeros((len(texts), fallback_dim))

# 独立的奖励函数（参照test.py的格式）
def travel_style_similarity_reward_func(similarity_model, prompts, completions, reference_responses, **kwargs) -> list[float]:
    """
    旅游风格相似度奖励函数

    Args:
        similarity_model: 用于计算相似度的模型实例
        prompts: 输入的prompt列表
        completions: 模型生成的completion列表
        reference_responses: 参考答案列表
        **kwargs: 其他可选参数

    Returns:
        list[float]: 奖励值列表
    """
    from sklearn.metrics.pairwise import cosine_similarity

    responses = [completion[0]['content'] if isinstance(completion, list) else completion for completion in completions]
    rewards = []

    print(f"\n{'='*80}")
    print(f"相似度奖励函数计算 - 处理 {len(responses)} 个生成结果")
    print(f"{'='*80}")

    for i, response in enumerate(responses):
        if i < len(reference_responses):
            reference = reference_responses[i]

            try:
                # 编码文本
                embeddings = similarity_model.encode([response, reference])

                # 计算余弦相似度
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

                # 确保相似度在0-1范围内
                similarity = max(0.0, min(1.0, similarity))

                # 将相似度转换为奖励分数
                reward = similarity
                rewards.append(reward)

                # 打印详细信息
                print(f"\n样本 {i+1}:")
                print(f"Prompt: {prompts[i][:100]}..." if len(prompts[i]) > 100 else f"Prompt: {prompts[i]}")
                print(f"生成回复: {response[:150]}..." if len(response) > 150 else f"生成回复: {response}")
                print(f"参考答案: {reference[:150]}..." if len(reference) > 150 else f"参考答案: {reference}")
                print(f"相似度: {similarity:.4f}")
                print(f"相似度奖励: {reward:.4f}")
                print(f"{'-'*60}")

            except Exception as e:
                logger.error(f"Error calculating similarity reward: {e}")
                rewards.append(0.0)
                print(f"样本 {i+1}: 相似度计算错误，奖励设为 0.0")
        else:
            rewards.append(0.0)
            print(f"样本 {i+1}: 缺少参考答案，奖励设为 0.0")

    avg_similarity_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"\n平均相似度奖励: {avg_similarity_reward:.4f}")
    print(f"{'='*80}\n")

    return rewards

def travel_style_length_reward_func(prompts, completions, reference_responses, **kwargs) -> list[float]:
    """
    旅游风格长度奖励函数

    Args:
        prompts: 输入的prompt列表
        completions: 模型生成的completion列表
        reference_responses: 参考答案列表
        **kwargs: 其他可选参数

    Returns:
        list[float]: 奖励值列表
    """
    responses = [completion[0]['content'] if isinstance(completion, list) else completion for completion in completions]
    rewards = []

    print(f"\n{'='*80}")
    print(f"长度奖励函数计算 - 处理 {len(responses)} 个生成结果")
    print(f"{'='*80}")

    for i, response in enumerate(responses):
        if i < len(reference_responses):
            reference = reference_responses[i]

            predicted_length = len(response.split())
            reference_length = len(reference.split())

            # 计算长度差异比率
            if reference_length > 0:
                length_diff = abs(predicted_length - reference_length) / reference_length
                # 长度越接近，奖励越高
                length_reward = max(0, 1 - length_diff) * 0.5
                rewards.append(length_reward)

                # 打印详细信息
                print(f"样本 {i+1}:")
                print(f"生成文本长度: {predicted_length} 词")
                print(f"参考文本长度: {reference_length} 词")
                print(f"长度差异比率: {length_diff:.4f}")
                print(f"长度奖励: {length_reward:.4f}")
                print(f"{'-'*60}")
            else:
                rewards.append(0.0)
                print(f"样本 {i+1}: 参考答案长度为0，奖励设为 0.0")
        else:
            rewards.append(0.0)
            print(f"样本 {i+1}: 缺少参考答案，奖励设为 0.0")

    avg_length_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"\n平均长度奖励: {avg_length_reward:.4f}")
    print(f"{'='*80}\n")

    return rewards

class TravelStyleGRPOTrainer:
    """使用GRPO方法训练旅游风格生成模型"""

    def __init__(self,
                 model_name: str = "../LLMs/Qwen3-8B",
                 similarity_model_name: str = "../LLMs/Qwen3-Embedding-4B",
                 device: str = "cuda",
                 use_lora: bool = True,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1,
                 use_accelerate: bool = False):
        """
        初始化GRPO训练器

        Args:
            model_name: 主模型名称
            device: 计算设备
            use_lora: 是否使用LoRA微调
            lora_r: LoRA的r参数
            lora_alpha: LoRA的alpha参数
            lora_dropout: LoRA的dropout率
            use_accelerate: 是否使用accelerate库
        """
        self.device = device
        self.model_name = model_name
        self.use_lora = use_lora
        self.use_accelerate = use_accelerate and ACCELERATE_AVAILABLE
        
        # 初始化accelerator
        if self.use_accelerate:
            self.accelerator = Accelerator()
        else:
            self.accelerator = None

        # 训练进度跟踪
        self.training_progress = None
        self.current_step = 0
        self.total_steps = 0

        from sentence_transformers import SentenceTransformer

        # 初始化相似度模型
        self.similarity_model = SentenceTransformer(similarity_model_name, device="cuda:2")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        if self.use_accelerate:
            # 使用accelerate时不设置device_map
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                max_memory={0: "20GiB", 1: "20GiB", 2: "0GiB"},
                torch_dtype=torch.bfloat16
            )

        # 配置LoRA
        if self.use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=lora_dropout,
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)

        logger.info("GRPO trainer initialized with LoRA and Accelerate" if (use_lora and self.use_accelerate) 
                   else "GRPO trainer initialized with LoRA" if use_lora 
                   else "GRPO trainer initialized")

    def prepare_dataset(self, text_dataset=None, save_dataset=True, batch_size=4) -> Dict:
        """
        准备训练数据集

        Args:
            text_dataset: TravelTextDataset实例，用于生成文本描述
            trajectories: 兼容性参数，包含(家乡轨迹, 目的地轨迹, 目的地名称)的元组列表
            save_dataset: 是否保存数据集到本地
            batch_size: 批处理大小，用于并行生成

        Returns:
            格式化的数据集和参考答案
        """
        queries = []
        reference_responses = []

        # 创建保存目录
        dataset_dir = "../dataset"
        if save_dataset:
            os.makedirs(dataset_dir, exist_ok=True)
            print(f"数据集将保存到: {dataset_dir}")

        if text_dataset is not None:
            # 使用新的文本数据集
            from data import TravelTextDataset

            print(f"\n{'='*80}")
            print(f"开始处理文本数据集 - 总共 {len(text_dataset)} 个样本")
            print(f"批处理大小: {batch_size} (支持多卡并行)")
            print(f"{'='*80}")

            # 生成LLM参考答案
            destination_generator = TravelStyleGenerator(self.model_name, self.device)

            # 收集所有需要处理的数据
            all_items = []
            for item in text_dataset:
                all_items.append(item)

            # 准备批处理数据
            hometown_prompts = []
            destination_prompts = []
            item_info = []

            for item in all_items:
                hometown_prompts.append(item['hometown_prompt'] + " /no_think")
                destination_prompts.append(item['destination_prompt'])
                item_info.append({
                    'uid': item['uid'],
                    'dst_region': item['dst_region'],
                    'ori_region': item['ori_region']
                })

            print(f"\n🔄 开始批量生成参考答案...")
            print(f"总样本数: {len(destination_prompts)}")
            print(f"预计批次数: {(len(destination_prompts) + batch_size - 1) // batch_size}")

            # 使用批处理生成参考答案
            batch_progress = tqdm(
                range(0, len(destination_prompts), batch_size),
                desc="批量生成参考答案",
                unit="batch",
                total=(len(destination_prompts) + batch_size - 1) // batch_size
            )

            successful_count = 0
            failed_count = 0

            for batch_start in batch_progress:
                batch_end = min(batch_start + batch_size, len(destination_prompts))
                batch_destination_prompts = destination_prompts[batch_start:batch_end]
                batch_item_info = item_info[batch_start:batch_end]

                # 更新进度条信息
                batch_progress.set_postfix({
                    '当前批次': f"{batch_start//batch_size + 1}/{(len(destination_prompts) + batch_size - 1) // batch_size}",
                    '批次大小': len(batch_destination_prompts),
                    '成功': successful_count,
                    '失败': failed_count
                })


                # 批量生成参考答案
                batch_references = destination_generator.get_output(batch_destination_prompts, max_length=512, temperature=0.7)

                # 处理批次结果
                for i, reference_style in enumerate(batch_references):
                    actual_index = batch_start + i
                    if reference_style != "Error in generation":
                        queries.append(hometown_prompts[actual_index])
                        reference_responses.append(reference_style)
                        successful_count += 1

                        # 详细日志（每50个样本打印一次）
                        if successful_count % 50 == 0:
                            current_info = batch_item_info[i]
                            print(f"\n📊 已成功处理 {successful_count} 个样本")
                            print(f"   当前样本 - 用户: {current_info['uid']}, 目标: {current_info['dst_region']}")
                            print(f"   Reference长度: {len(reference_style)} 字符")
                            print(f"   Reference预览: {reference_style[:100]}...")
                            print(f"{'-'*60}")
                    else:
                        failed_count += 1
                        current_info = batch_item_info[i]
                        logger.warning(f"Failed to generate reference for user {current_info['uid']}")

                # 更新最终进度
                batch_progress.set_postfix({
                    '当前批次': f"{batch_start//batch_size + 1}/{(len(destination_prompts) + batch_size - 1) // batch_size}",
                    '成功': successful_count,
                    '失败': failed_count,
                    '成功率': f"{(successful_count/(successful_count+failed_count))*100:.1f}%" if (successful_count+failed_count) > 0 else "0%"
                })

            batch_progress.close()

            print(f"\n✅ 文本数据集处理完成!")
            print(f"   总样本数: {len(text_dataset)}")
            print(f"   成功处理: {successful_count}")
            print(f"   失败数量: {failed_count}")
            print(f"   成功率: {(successful_count/(successful_count+failed_count))*100:.1f}%")
            print(f"   批处理效率: 平均每批次处理 {successful_count/((len(destination_prompts) + batch_size - 1) // batch_size):.1f} 个样本")
        else:
            raise ValueError("Either text_dataset or trajectories must be provided")

        # 创建数据集
        from datasets import Dataset
        dataset = Dataset.from_dict({
            "prompt": queries,
            "reference": reference_responses
        })

        print(f"\n📦 创建数据集完成!")
        print(f"   最终数据集大小: {len(dataset)}")
        print(f"   Prompt样本数: {len(queries)}")
        print(f"   Reference样本数: {len(reference_responses)}")

        # 保存数据集
        if save_dataset and len(dataset) > 0:
            try:
                print(f"\n💾 保存数据集到本地...")

                # 生成时间戳文件名
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                dataset_name = f"travel_dataset_{timestamp}"
                dataset_path = os.path.join(dataset_dir, dataset_name)

                # 保存dataset
                dataset.save_to_disk(dataset_path)

                # 保存额外的元数据
                metadata = {
                    "dataset_size": len(dataset),
                    "creation_time": timestamp,
                    "model_name": self.model_name,
                    "successful_samples": successful_count,
                    "failed_samples": failed_count,
                    "success_rate": (successful_count/(successful_count+failed_count))*100 if (successful_count+failed_count) > 0 else 0,
                    "batch_size": batch_size,
                    "processing_mode": "batch_parallel"
                }

                metadata_path = os.path.join(dataset_path, "metadata.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                print(f"✅ 数据集保存成功!")
                print(f"   保存路径: {dataset_path}")
                print(f"   元数据文件: {metadata_path}")

                # 保存样本预览
                sample_path = os.path.join(dataset_path, "sample_preview.json")
                sample_data = []
                for i in range(min(3, len(dataset))):
                    sample_data.append({
                        "index": i,
                        "prompt": dataset[i]["prompt"][:200] + "..." if len(dataset[i]["prompt"]) > 200 else dataset[i]["prompt"],
                        "reference": dataset[i]["reference"][:200] + "..." if len(dataset[i]["reference"]) > 200 else dataset[i]["reference"]
                    })

                with open(sample_path, 'w', encoding='utf-8') as f:
                    json.dump(sample_data, f, indent=2, ensure_ascii=False)

                print(f"   样本预览: {sample_path}")

            except Exception as e:
                logger.error(f"Failed to save dataset: {e}")
                print(f"❌ 数据集保存失败: {e}")

        print(f"\n{'='*80}")
        print(f"数据集准备完成!")
        print(f"使用批处理模式，批次大小: {batch_size}")
        print(f"{'='*80}\n")

        return dataset, reference_responses

    def train(self, text_dataset=None,
              output_dir: str = "./grpo_travel_style_model",
              run_name: str = "travel_style_grpo",
              num_train_epochs: int = 1,
              learning_rate: float = 5e-5,
              per_device_train_batch_size: int = 2,
              gradient_accumulation_steps: int = 2,
              num_generations: int = 4,
              max_prompt_length: int = 3072,
              max_completion_length: int = 256,
              save_steps: int = 250,
              logging_steps: int = 1):
        """
        使用GRPO训练模型

        Args:
            text_dataset: datasets实例，用于文本描述训练
            output_dir: 输出目录
            run_name: 运行名称
            num_train_epochs: 训练轮数
            learning_rate: 学习率
            per_device_train_batch_size: 每个设备的批次大小
            gradient_accumulation_steps: 梯度累积步数
            num_generations: 每个prompt生成的样本数
            max_prompt_length: 最大prompt长度
            max_completion_length: 最大completion长度
            save_steps: 保存步数
            logging_steps: 日志步数
        """
        print(f"\n{'='*100}")
        print(f"开始GRPO强化学习训练")
        print(f"{'='*100}")

        # 准备数据集
        print("准备训练数据集...")
        dataset = text_dataset

        # 计算总步数
        dataset_size = len(dataset)
        self.total_steps = (dataset_size * num_train_epochs) // (per_device_train_batch_size * gradient_accumulation_steps)

        print(f"数据集大小: {dataset_size}")
        print(f"训练轮数: {num_train_epochs}")
        print(f"批次大小: {per_device_train_batch_size}")
        print(f"梯度累积步数: {gradient_accumulation_steps}")
        print(f"预计总训练步数: {self.total_steps}")
        print(f"输出目录: {output_dir}")

        # 初始化进度条
        self.training_progress = tqdm(
            total=self.total_steps,
            desc="GRPO训练进度",
            unit="step",
            position=0,
            leave=True
        )

        # 初始化SwanLab实验跟踪
        if SWANLAB_AVAILABLE:
            config_dict = {
                "model_name": self.model_name,
                "use_lora": self.use_lora,
                "num_train_epochs": num_train_epochs,
                "learning_rate": learning_rate,
                "per_device_train_batch_size": per_device_train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "num_generations": num_generations,
                "max_prompt_length": max_prompt_length,
                "max_completion_length": max_completion_length,
                "dataset_size": dataset_size,
                "total_steps": self.total_steps,
            }
            if text_dataset:
                config_dict["num_trajectories"] = len(text_dataset)

            swanlab.init(
                project="travel-style-grpo",
                experiment_name=run_name,
                config=config_dict
            )
            logger.info("SwanLab experiment initialized")
        else:
            logger.warning("SwanLab not available, metrics will not be logged")

        # 配置训练参数
        training_args = GRPOConfig(
            output_dir=output_dir,
            # 禁用wandb相关功能
            report_to=None,  # 不报告到任何平台
            run_name=run_name,
            learning_rate=learning_rate,
            adam_beta1=0.9,
            adam_beta2=0.99,
            # weight_decay=0.1,
            # warmup_ratio=0.1,
            lr_scheduler_type='cosine',
            logging_steps=logging_steps,
            bf16=True,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_generations=num_generations,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            num_train_epochs=num_train_epochs,
            save_steps=save_steps,
            max_grad_norm=0.1,
            log_on_each_node=False,
            use_vllm=False,
        )

        # 定义奖励函数，传入reference_responses和进度跟踪作为闭包变量
        def similarity_reward_func(prompts, completions, reference, **kwargs):
            # 更新进度条
            self.current_step += 1
            self.training_progress.update(1)
            self.training_progress.set_postfix({
                'Step': f"{self.current_step}/{self.total_steps}",
                'Phase': '相似度奖励计算'
            })

            print(f"\n🔄 步骤 {self.current_step}/{self.total_steps}: 相似度奖励计算")
            similarity_rewards = travel_style_similarity_reward_func(self.similarity_model, prompts, completions, reference, **kwargs)
            # 记录到SwanLab
            if SWANLAB_AVAILABLE:
                swanlab.log({
                    "step": self.current_step,
                    "reward/similarity": sum(similarity_rewards) / len(similarity_rewards) if similarity_rewards else 0.0,
                    "progress": (self.current_step/self.total_steps)*100
                })
            return similarity_rewards

        def length_reward_func(prompts, completions, reference, **kwargs):
            self.training_progress.set_postfix({
                'Step': f"{self.current_step}/{self.total_steps}",
                'Phase': '长度奖励计算'
            })

            print(f"\n📏 步骤 {self.current_step}/{self.total_steps}: 长度奖励计算")
            return travel_style_length_reward_func(prompts, completions, reference, **kwargs)

        def quality_reward_func(prompts, completions, reference, **kwargs):
            self.training_progress.set_postfix({
                'Step': f"{self.current_step}/{self.total_steps}",
                'Phase': '质量奖励计算'
            })

            # 计算并显示总奖励
            similarity_rewards = travel_style_similarity_reward_func(self.similarity_model, prompts, completions, reference, **kwargs)
            length_rewards = travel_style_length_reward_func(prompts, completions, reference, **kwargs)

            total_rewards = [s + l for s, l in zip(similarity_rewards, length_rewards)]
            avg_total_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0

            print(f"\n🎯 步骤 {self.current_step}/{self.total_steps} 总结:")
            print(f"   平均总奖励: {avg_total_reward:.4f}")
            print(f"   完成进度: {(self.current_step/self.total_steps)*100:.1f}%")

            # 记录到SwanLab
            if SWANLAB_AVAILABLE:
                swanlab.log({
                    "step": self.current_step,
                    "reward/similarity": sum(similarity_rewards) / len(similarity_rewards) if similarity_rewards else 0.0,
                    "reward/length": sum(length_rewards) / len(length_rewards) if length_rewards else 0.0,
                    "reward/total": avg_total_reward,
                    "progress": (self.current_step/self.total_steps)*100
                })

            return total_rewards

        # 创建GRPO训练器
        if self.use_accelerate:
            # 使用accelerate准备模型和数据集
            model, dataset = self.accelerator.prepare(self.model, dataset)
            trainer = GRPOTrainer(
                model=model,
                processing_class=self.tokenizer,
                reward_funcs=[
                    similarity_reward_func,
                ],
                args=training_args,
                train_dataset=dataset,
            )
        else:
            trainer = GRPOTrainer(
                model=self.model,
                processing_class=self.tokenizer,
                reward_funcs=[
                    similarity_reward_func,
                ],
                args=training_args,
                train_dataset=dataset,
            )

        logger.info("Starting GRPO training...")
        print(f"\n🚀 开始强化学习训练...")
        print(f"模型: {self.model_name}")
        print(f"使用LoRA: {'是' if self.use_lora else '否'}")
        if self.use_accelerate:
            print(f"使用 accelerate: 是")

        # 记录训练开始时间
        start_time = time.time()
        if SWANLAB_AVAILABLE:
            swanlab.log({"training_status": "started", "start_time": start_time})

        try:
            trainer.train()
        except Exception as e:
            print(f"\n❌ 训练过程中出现错误: {e}")
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # 确保进度条关闭
            if self.training_progress:
                self.training_progress.close()

        # 计算训练时间
        end_time = time.time()
        training_duration = end_time - start_time

        print(f"\n✅ 训练完成!")
        print(f"训练用时: {training_duration:.2f} 秒 ({training_duration/60:.1f} 分钟)")
        print(f"总步数: {self.current_step}")

        # 记录训练完成
        if SWANLAB_AVAILABLE:
            swanlab.log({
                "training_status": "completed",
                "end_time": end_time,
                "training_duration": training_duration,
                "final_step": self.current_step
            })

        # 保存模型
        print(f"\n💾 保存模型到: {output_dir}")
        trainer.save_model(output_dir)
        logger.info(f"Model saved to {output_dir}")

        # 记录模型保存信息
        if SWANLAB_AVAILABLE:
            swanlab.log({
                "model_saved": True,
                "model_path": output_dir,
                "final_epoch": num_train_epochs
            })

        print(f"\n{'='*100}")
        print(f"GRPO强化学习训练完成!")
        print(f"{'='*100}\n")

def main():
    """主函数 - 演示使用方法"""
    import wandb
    import argparse
    wandb.init(mode="disabled")  # 强制禁用 wandb
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_accelerate", action="store_true", help="是否使用accelerate加速")
    args = parser.parse_args()
    # 初始化训练器（使用LoRA和accelerate）
    trainer = TravelStyleGRPOTrainer(
        model_name="../LLMs/Qwen3-8B",
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        use_accelerate=args.use_accelerate  # 启用accelerate支持
    )

    # 准备训练和测试数据
    from datasets import load_from_disk
    text_dataset = load_from_disk("../dataset/travel_dataset_20250708_193154")

    # 训练模型
    trainer.train(
        text_dataset=text_dataset,
        output_dir="./grpo_travel_style_lora_model",
        run_name="travel_style_grpo_lora",
        num_train_epochs=10,
        per_device_train_batch_size=1,  # 减小批次大小适应示例数据
        gradient_accumulation_steps=8
    )

    # 评估模型
    # results = trainer.evaluate(test_trajectories)
    # print(f"Evaluation results: {results}")
    
    # 结束SwanLab实验记录
    if SWANLAB_AVAILABLE:
        swanlab.finish()
        print("SwanLab experiment finished")

if __name__ == "__main__":
    main()
