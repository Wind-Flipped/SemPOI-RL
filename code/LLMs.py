"""
LLMs.py - 大语言模型调用接口和强化学习训练模块

该模块包含：
1. 调用大语言模型生成旅游风格的接口
2. 使用GRPO方法对LLM进行强化学习的训练接口
3. 文本相似度计算和奖励函数
"""

import os
# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

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
    
    def __init__(self, model_name: str = "/home/liuyq/X-R1/Qwen3-8B", device: str = "cuda"):
        """
        初始化旅游风格生成器
        
        Args:
            model_name: 预训练模型名称
            device: 计算设备
        """
        self.device = device
        self.model_name = model_name
        
        # 初始化提示格式化器
        self.prompt_formatter = TravelPromptFormatter()
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        logger.info(f"Travel style generator initialized with {model_name}")
    
    def generate_travel_style(self, trajectory: TravelTrajectory, target_region: str, 
                            max_length: int = 150, temperature: float = 0.7,
                            prompt_type: str = "basic") -> str:
        """
        生成旅游风格描述
        
        Args:
            trajectory: 旅游轨迹
            target_region: 目标地区
            max_length: 最大生成长度
            temperature: 生成温度
            prompt_type: 提示类型 ('basic', 'detailed')
            
        Returns:
            生成的旅游风格描述
        """
        # 使用prompt模块生成提示
        prompt_content = self.prompt_formatter.trajectory_to_prompt(trajectory, target_region, prompt_type)
        
        try:
            # 构建chat格式的消息
            messages = [
                {"role": "user", "content": prompt_content}
            ]
            
            # 使用chat模板生成文本，禁用思考模式
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # 禁用Qwen3的思考模式
            )
            
            # 编码输入
            inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.model.device)
            
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # 解码生成的文本
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的部分（移除原始输入）
            travel_style = generated_text[len(text):].strip()
            
            return travel_style
            
        except Exception as e:
            logger.error(f"Error generating travel style: {e}")
            return "Error in generation"
    
    def generate_style_from_prompt(self, destination_prompt: str, 
                                 max_length: int = 150, temperature: float = 0.7) -> str:
        """
        从destination prompt直接生成旅游风格描述
        
        Args:
            destination_prompt: 包含目的地轨迹信息的提示文本
            max_length: 最大生成长度
            temperature: 生成温度
            
        Returns:
            生成的旅游风格描述
        """
        try:
            # 构建chat格式的消息
            messages = [
                {"role": "user", "content": destination_prompt}
            ]
            
            # 使用chat模板生成文本，禁用思考模式
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # 禁用Qwen3的思考模式
            )
            
            # 编码输入
            inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.model.device)
            
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # 解码生成的文本
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的部分（移除原始输入）
            travel_style = generated_text[len(text):].strip()
            
            return travel_style
            
        except Exception as e:
            logger.error(f"Error generating travel style from prompt: {e}")
            return "Error in generation"
        
class TravelStyleRewardCalculator:
    """旅游风格相似度计算和奖励函数"""
    
    def __init__(self, similarity_model: str = "/home/liuyq/X-R1/Qwen3-Embedding-0.6B"):
        """
        初始化奖励计算器
        
        Args:
            similarity_model: 用于计算文本相似度的模型
        """
        self.similarity_model = SentenceTransformer(similarity_model)
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

# 独立的奖励函数（参照test.py的格式）
def travel_style_similarity_reward_func(prompts, completions, reference_responses, **kwargs) -> list[float]:
    """
    旅游风格相似度奖励函数
    
    Args:
        prompts: 输入的prompt列表
        completions: 模型生成的completion列表
        reference_responses: 参考答案列表
        **kwargs: 其他可选参数
        
    Returns:
        list[float]: 奖励值列表
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 初始化相似度模型
    similarity_model = SentenceTransformer("/home/liuyq/X-R1/Qwen3-Embedding-0.6B")
    
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
                reward = similarity * 2.0  # 放大奖励范围
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

def travel_style_content_quality_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    旅游风格内容质量奖励函数
    检查生成的旅游风格描述是否包含关键要素
    
    Args:
        prompts: 输入的prompt列表
        completions: 模型生成的completion列表
        **kwargs: 其他可选参数
        
    Returns:
        list[float]: 奖励值列表
    """
    responses = [completion[0]['content'] if isinstance(completion, list) else completion for completion in completions]
    rewards = []
    
    # 定义旅游风格关键词
    key_elements = [
        ['attraction', 'museum', 'park', 'landmark', 'site'],  # 景点类型
        ['activity', 'walking', 'sightseeing', 'shopping', 'dining'],  # 活动类型
        ['pace', 'schedule', 'time', 'leisurely', 'fast'],  # 节奏相关
        ['prefer', 'like', 'enjoy', 'interest', 'style']  # 偏好相关
    ]
    
    print(f"\n{'='*80}")
    print(f"内容质量奖励函数计算 - 处理 {len(responses)} 个生成结果")
    print(f"{'='*80}")
    
    for idx, response in enumerate(responses):
        response_lower = response.lower()
        quality_score = 0.0
        quality_details = []
        
        # 检查是否包含各类关键要素
        for i, element_group in enumerate(key_elements):
            group_names = ['景点类型', '活动类型', '节奏相关', '偏好相关']
            if any(keyword in response_lower for keyword in element_group):
                quality_score += 0.25
                found_keywords = [kw for kw in element_group if kw in response_lower]
                quality_details.append(f"{group_names[i]}: {found_keywords}")
        
        # 检查长度是否合适（20-100词）
        word_count = len(response.split())
        if 20 <= word_count <= 100:
            quality_score += 0.5
            quality_details.append(f"长度合适: {word_count}词")
        else:
            quality_details.append(f"长度不合适: {word_count}词 (建议20-100词)")
        
        # 检查是否包含具体描述（避免过于抽象）
        sentence_count = len(response.split('.'))
        if sentence_count >= 2:  # 至少两个句子
            quality_score += 0.25
            quality_details.append(f"句子数量充足: {sentence_count}句")
        else:
            quality_details.append(f"句子数量不足: {sentence_count}句")
        
        rewards.append(quality_score)
        
        # 打印详细信息
        print(f"\n样本 {idx+1}:")
        print(f"生成回复: {response[:200]}..." if len(response) > 200 else f"生成回复: {response}")
        print(f"质量评分细节:")
        for detail in quality_details:
            print(f"  - {detail}")
        print(f"总质量奖励: {quality_score:.4f}")
        print(f"{'-'*60}")
    
    avg_quality_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"\n平均质量奖励: {avg_quality_reward:.4f}")
    print(f"{'='*80}\n")
    
    return rewards

class TravelStyleGRPOTrainer:
    """使用GRPO方法训练旅游风格生成模型"""
    
    def __init__(self, 
                 model_name: str = "/home/liuyq/X-R1/Qwen3-8B",
                 device: str = "cuda",
                 use_lora: bool = True,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1):
        """
        初始化GRPO训练器
        
        Args:
            model_name: 主模型名称
            device: 计算设备
            use_lora: 是否使用LoRA微调
            lora_r: LoRA的r参数
            lora_alpha: LoRA的alpha参数
            lora_dropout: LoRA的dropout率
        """
        self.device = device
        self.model_name = model_name
        self.use_lora = use_lora
        
        # 训练进度跟踪
        self.training_progress = None
        self.current_step = 0
        self.total_steps = 0
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
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
        
        # 初始化奖励计算器
        self.reward_calculator = TravelStyleRewardCalculator()
        
        logger.info("GRPO trainer initialized with LoRA" if use_lora else "GRPO trainer initialized")
    
    def prepare_dataset(self, text_dataset=None, trajectories=None, save_dataset=True) -> Dict:
        """
        准备训练数据集
        
        Args:
            text_dataset: TravelTextDataset实例，用于生成文本描述
            trajectories: 兼容性参数，包含(家乡轨迹, 目的地轨迹, 目的地名称)的元组列表
            save_dataset: 是否保存数据集到本地
            
        Returns:
            格式化的数据集和参考答案
        """
        queries = []
        reference_responses = []
        
        # 创建保存目录
        dataset_dir = "/home/liuyq/2025summer/SPOT-Trip/dataset"
        if save_dataset:
            os.makedirs(dataset_dir, exist_ok=True)
            print(f"数据集将保存到: {dataset_dir}")
        
        if text_dataset is not None:
            # 使用新的文本数据集
            from data import TravelTextDataset
            
            print(f"\n{'='*80}")
            print(f"开始处理文本数据集 - 总共 {len(text_dataset)} 个样本")
            print(f"{'='*80}")
            
            # 生成LLM参考答案
            destination_generator = TravelStyleGenerator(self.model_name, self.device)
            
            # 添加进度条
            progress_bar = tqdm(
                text_dataset, 
                desc="处理文本数据集", 
                unit="样本",
                total=len(text_dataset),
                position=0,
                leave=True
            )
            
            successful_count = 0
            failed_count = 0
            
            for idx, item in enumerate(progress_bar):
                hometown_prompt = item['hometown_prompt']
                destination_prompt = item['destination_prompt']
                dst_region = item['dst_region']
                uid = item['uid']
                
                # 更新进度条描述
                progress_bar.set_postfix({
                    '成功': successful_count,
                    '失败': failed_count,
                    '当前用户': uid,
                    '目标城市': dst_region
                })
                
                # 使用LLM处理destination_prompt生成reference
                try:
                    # 提取destination_prompt中的轨迹信息并生成旅游风格描述
                    reference_style = destination_generator.generate_style_from_prompt(destination_prompt)
                    
                    queries.append(hometown_prompt + " /no_think")
                    reference_responses.append(reference_style)
                    successful_count += 1
                    
                    # 打印详细处理信息（每10个样本打印一次）
                    if (idx + 1) % 10 == 0:
                        print(f"\n📊 已处理 {idx + 1}/{len(text_dataset)} 个样本")
                        print(f"   成功: {successful_count}, 失败: {failed_count}")
                        print(f"   当前样本 - 用户: {uid}, 目标: {dst_region}")
                        print(f"   Hometown prompt长度: {len(hometown_prompt)} 字符")
                        print(f"   生成reference长度: {len(reference_style)} 字符")
                        print(f"   Reference预览: {reference_style[:100]}...")
                        print(f"{'-'*60}")
                        
                except Exception as e:
                    failed_count += 1
                    logger.warning(f"Failed to generate reference for user {uid}: {e}")
                    progress_bar.set_postfix({
                        '成功': successful_count,
                        '失败': failed_count,
                        '当前用户': uid,
                        '错误': str(e)[:20] + "..."
                    })
                    continue
            
            progress_bar.close()
            
            print(f"\n✅ 文本数据集处理完成!")
            print(f"   总样本数: {len(text_dataset)}")
            print(f"   成功处理: {successful_count}")
            print(f"   失败数量: {failed_count}")
            print(f"   成功率: {(successful_count/(successful_count+failed_count))*100:.1f}%")
        
        elif trajectories is not None:
            # 兼容原有的轨迹数据格式
            print(f"\n{'='*80}")
            print(f"开始处理轨迹数据 - 总共 {len(trajectories)} 个轨迹")
            print(f"{'='*80}")
            
            prompt_formatter = TravelPromptFormatter()
            
            # 添加进度条
            progress_bar = tqdm(
                trajectories, 
                desc="处理轨迹数据", 
                unit="轨迹",
                total=len(trajectories),
                position=0,
                leave=True
            )
            
            successful_count = 0
            failed_count = 0
            
            for idx, (hometown_traj, destination_traj, destination_name) in enumerate(progress_bar):
                # 更新进度条描述
                progress_bar.set_postfix({
                    '成功': successful_count,
                    '失败': failed_count,
                    '目标': destination_name
                })
                
                try:
                    # 生成家乡轨迹的提示
                    prompt = prompt_formatter.trajectory_to_prompt(hometown_traj, destination_name)
                    
                    # 生成目的地实际风格（作为参考）
                    destination_generator = TravelStyleGenerator(self.model_name, self.device)
                    actual_style = destination_generator.generate_travel_style(destination_traj, destination_name)
                    
                    queries.append(prompt + " /no_think")
                    reference_responses.append(actual_style)
                    successful_count += 1
                    
                    # 打印详细处理信息（每5个样本打印一次）
                    if (idx + 1) % 5 == 0:
                        print(f"\n📊 已处理 {idx + 1}/{len(trajectories)} 个轨迹")
                        print(f"   成功: {successful_count}, 失败: {failed_count}")
                        print(f"   当前目标: {destination_name}")
                        print(f"   Prompt长度: {len(prompt)} 字符")
                        print(f"   生成style长度: {len(actual_style)} 字符")
                        print(f"{'-'*60}")
                        
                except Exception as e:
                    failed_count += 1
                    logger.warning(f"Failed to process trajectory for {destination_name}: {e}")
                    progress_bar.set_postfix({
                        '成功': successful_count,
                        '失败': failed_count,
                        '错误': str(e)[:20] + "..."
                    })
                    continue
            
            progress_bar.close()
            
            print(f"\n✅ 轨迹数据处理完成!")
            print(f"   总轨迹数: {len(trajectories)}")
            print(f"   成功处理: {successful_count}")
            print(f"   失败数量: {failed_count}")
            print(f"   成功率: {(successful_count/(successful_count+failed_count))*100:.1f}%")
        
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
                    "success_rate": (successful_count/(successful_count+failed_count))*100 if (successful_count+failed_count) > 0 else 0
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
        print(f"{'='*80}\n")
        
        return dataset, reference_responses
    
    def train(self, text_dataset=None, trajectories=None, 
              output_dir: str = "./grpo_travel_style_model",
              run_name: str = "travel_style_grpo",
              num_train_epochs: int = 1,
              learning_rate: float = 5e-6,
              per_device_train_batch_size: int = 2,
              gradient_accumulation_steps: int = 2,
              num_generations: int = 4,
              max_prompt_length: int = 256,
              max_completion_length: int = 256,
              save_steps: int = 100,
              logging_steps: int = 1):
        """
        使用GRPO训练模型
        
        Args:
            text_dataset: TravelTextDataset实例，用于文本描述训练
            trajectories: 兼容性参数，轨迹数据
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
        dataset, reference_responses = self.prepare_dataset(text_dataset=text_dataset, trajectories=trajectories)
        
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
            if trajectories:
                config_dict["num_trajectories"] = len(trajectories)
            
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
            weight_decay=0.1,
            warmup_ratio=0.1,
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
        def similarity_reward_func(prompts, completions, **kwargs):
            # 更新进度条
            self.current_step += 1
            self.training_progress.update(1)
            self.training_progress.set_postfix({
                'Step': f"{self.current_step}/{self.total_steps}",
                'Phase': '相似度奖励计算'
            })
            
            print(f"\n🔄 步骤 {self.current_step}/{self.total_steps}: 相似度奖励计算")
            return travel_style_similarity_reward_func(prompts, completions, reference_responses, **kwargs)
        
        def length_reward_func(prompts, completions, **kwargs):
            self.training_progress.set_postfix({
                'Step': f"{self.current_step}/{self.total_steps}",
                'Phase': '长度奖励计算'
            })
            
            print(f"\n📏 步骤 {self.current_step}/{self.total_steps}: 长度奖励计算")
            return travel_style_length_reward_func(prompts, completions, reference_responses, **kwargs)
        
        def quality_reward_func(prompts, completions, **kwargs):
            self.training_progress.set_postfix({
                'Step': f"{self.current_step}/{self.total_steps}",
                'Phase': '质量奖励计算'
            })
            
            print(f"\n⭐ 步骤 {self.current_step}/{self.total_steps}: 质量奖励计算")
            rewards = travel_style_content_quality_reward_func(prompts, completions, **kwargs)
            
            # 计算并显示总奖励
            similarity_rewards = travel_style_similarity_reward_func(prompts, completions, reference_responses, **kwargs)
            length_rewards = travel_style_length_reward_func(prompts, completions, reference_responses, **kwargs)
            
            total_rewards = [s + l + q for s, l, q in zip(similarity_rewards, length_rewards, rewards)]
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
                    "reward/quality": sum(rewards) / len(rewards) if rewards else 0.0,
                    "reward/total": avg_total_reward,
                    "progress": (self.current_step/self.total_steps)*100
                })
            
            return rewards
        
        # 创建GRPO训练器
        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[
                similarity_reward_func,
                length_reward_func,
                quality_reward_func,
            ],
            args=training_args,
            train_dataset=dataset,
        )
        
        logger.info("Starting GRPO training...")
        print(f"\n🚀 开始强化学习训练...")
        print(f"模型: {self.model_name}")
        print(f"使用LoRA: {'是' if self.use_lora else '否'}")
        
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
    
    def evaluate(self, test_trajectories: List[Tuple[TravelTrajectory, TravelTrajectory, str]]) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            test_trajectories: 测试轨迹数据
            
        Returns:
            评估指标
        """
        total_reward = 0.0
        total_similarity = 0.0
        num_samples = len(test_trajectories)
        
        generator = TravelStyleGenerator(self.model_name, self.device)
        # 如果使用了LoRA训练，需要将训练后的模型设置给generator
        if hasattr(self, 'model'):
            generator.model = self.model
        
        for hometown_traj, destination_traj, destination_name in test_trajectories:
            # 生成预测风格
            predicted_style = generator.generate_travel_style(hometown_traj, destination_name)
            
            # 生成实际风格作为参考
            actual_style = generator.generate_travel_style(destination_traj, destination_name)
            
            # 计算奖励和相似度
            reward = self.reward_calculator.calculate_reward(predicted_style, actual_style)
            similarity = self.reward_calculator.calculate_similarity(predicted_style, actual_style)
            
            total_reward += reward
            total_similarity += similarity
        
        results = {
            "average_reward": total_reward / num_samples,
            "average_similarity": total_similarity / num_samples,
            "num_samples": num_samples
        }
        
        # 记录评估结果到SwanLab
        if SWANLAB_AVAILABLE:
            swanlab.log({
                "eval/average_reward": results["average_reward"],
                "eval/average_similarity": results["average_similarity"],
                "eval/num_samples": results["num_samples"]
            })
            logger.info("Evaluation results logged to SwanLab")
        
        return results

# 使用示例和工具函数
def create_sample_trajectory(user_id: str, region: str) -> TravelTrajectory:
    """创建示例轨迹数据"""
    if region == "hometown":
        return TravelTrajectory(
            user_id=user_id,
            locations=["Central Park", "Local Museum", "Shopping Mall"],
            timestamps=["09:00", "14:00", "18:00"],
            activities=["walking", "sightseeing", "shopping"],
            region="hometown"
        )
    else:
        return TravelTrajectory(
            user_id=user_id,
            locations=["Eiffel Tower", "Louvre Museum", "Champs-Élysées"],
            timestamps=["10:00", "15:00", "19:00"],
            activities=["sightseeing", "culture", "shopping"],
            region="destination"
        )

def main():
    """主函数 - 演示使用方法"""
    import wandb
    wandb.init(mode="disabled")  # 强制禁用 wandb
    # 创建示例数据
    sample_trajectories = []
    for i in range(5):
        hometown_traj = create_sample_trajectory(f"user_{i}", "hometown")
        destination_traj = create_sample_trajectory(f"user_{i}", "destination")
        sample_trajectories.append((hometown_traj, destination_traj, "Paris"))
    
    # 初始化训练器（使用LoRA）
    trainer = TravelStyleGRPOTrainer(
        model_name="/home/liuyq/X-R1/Qwen3-8B",
        use_lora=True,
        lora_r=16,
        lora_alpha=32
    )
    
    # 准备训练和测试数据
    train_trajectories = sample_trajectories[:3]
    test_trajectories = sample_trajectories[3:]
    
    # 训练模型
    trainer.train(
        trajectories=train_trajectories,
        output_dir="./grpo_travel_style_lora_model",
        run_name="travel_style_grpo_lora",
        num_train_epochs=1,
        per_device_train_batch_size=2,  # 减小批次大小适应示例数据
        gradient_accumulation_steps=2
    )
    
    # 评估模型
    results = trainer.evaluate(test_trajectories)
    print(f"Evaluation results: {results}")
    
    # 结束SwanLab实验记录
    if SWANLAB_AVAILABLE:
        swanlab.finish()
        print("SwanLab experiment finished")

if __name__ == "__main__":
    main()
