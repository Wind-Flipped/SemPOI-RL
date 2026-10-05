# -*- coding: UTF-8 -*-
"""
LLMs.py - Interfaces for large language models and reinforcement learning utilities.

This module provides:
1. Interfaces for generating travel-style narratives with large language models.
2. GRPO-based reinforcement learning training utilities for LLMs.
3. Text similarity metrics and reward functions.
"""

import os

import torch
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
from tqdm import tqdm
import time
# from vllm import LLM, SamplingParams
import logging

# Import accelerate for distributed training
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

# Import SwanLab for experiment tracking
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    logging.warning("SwanLab not available. Install with: pip install swanlab")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TravelStyleGenerator:
    """Travel style generator that wraps LLM inference."""
    
    def __init__(self, model_name: str = "../LLMs/Qwen3-8B", device: str = "cuda",
                 use_vllm=False, use_lora=False, lora_path: str = None,
                 lora_path2=None):
        """
        Initialize the travel style generator.

        Args:
            model_name: Name of the pretrained model to load.
            device: Target device for inference.
            use_lora: Whether to load LoRA weights.
            lora_path: Path to the LoRA checkpoint.
        """
        self.device = device
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.use_lora = use_lora
        self.lora_path = lora_path
        if use_lora and not lora_path:
            raise ValueError("lora_path is required when use_lora=True")
        if use_vllm and use_lora:
            raise ValueError(
                "The vLLM backend does not load this project's PEFT adapters. "
                "Use the Transformers backend for SFT/RL profile exports."
            )
        self.model_stage = "base"
        # Silence vLLM logging
        logging.getLogger("vllm").setLevel(logging.CRITICAL)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if use_vllm:
            self.sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=512)
            self.model = LLM(model=model_name, max_model_len=2048, tensor_parallel_size=2,
                             max_num_seqs=4, gpu_memory_utilization=0.8)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            for param in self.model.parameters():
                param.requires_grad = False  # freeze the model - train adapters later

            if use_lora:
                try:
                    from peft import PeftModel
                    self.model = PeftModel.from_pretrained(self.model, lora_path, is_trainable=False)
                    self.model_stage = "sft"
                    # 2 Lora configs
                    if lora_path2 is not None:
                        self.model = PeftModel.from_pretrained(self.model, lora_path2, is_trainable=False)
                        self.model_stage = "rl"
                    logger.info(f"LoRA parameters loaded from {lora_path} and merged with base model.")
                except Exception as e:
                    logger.error(f"Failed to load LoRA parameters: {e}")
                    raise RuntimeError(
                        "Failed to load the requested LoRA checkpoint; refusing to "
                        "continue with a mislabeled Base/SFT/RL run."
                    ) from e

        logger.info(f"Travel style generator initialized with {model_name}")

    def get_output(self, messages: List[str], max_length: int = 256,
                   temperature: float = 0.7, batch_size: int = 4) -> List[str]:
        """
        Generate batched text outputs.

        Args:
            messages: List of input prompts.
            max_length: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            batch_size: Batch size for generation.

        Returns:
            List of generated texts.
        """
        results = []

        # Process in batches
        for i in range(0, len(messages), batch_size):
            batch_messages = messages[i:i + batch_size]
            batch_results = self._generate_batch_from_messages(batch_messages, max_length, temperature)
            results.extend(batch_results)

        return results

    def _generate_batch_from_messages(self, messages: List[str], max_length: int, temperature: float) -> List[str]:
        """
        Internal helper to generate outputs for a batch of raw prompts.

        Args:
            messages: List of user prompts.
            max_length: Maximum number of new tokens to generate.
            temperature: Sampling temperature.

        Returns:
            List of generated strings.
        """
        try:
            # Build chat-formatted prompts
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
                # Use vLLM for batched generation
                outputs = self.model.generate(
                    all_texts,
                    sampling_params=self.sampling_params
                )
                generated_texts = [output.outputs[0].text for output in outputs]
                return generated_texts

            # Batch encode with padding to support variable-length inputs
            inputs = self.tokenizer(
                all_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            ).to(self.model.device)

            input_lengths = inputs['attention_mask'].sum(dim=1)  # Actual length of each input (excluding padding)

            # Batched generation (leverages model-level parallelism)
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

            # Build masks to extract the generated portion

            outputs_only_generated = []
            for i, output in enumerate(outputs):
                input_len = input_lengths[i]
                output = outputs[i]
                outputs_only_generated.append(output[input_len:])

            # Decode the generated segments
            generated_texts = self.tokenizer.batch_decode(
                outputs_only_generated,
                skip_special_tokens=True
            )
            return generated_texts

        except Exception as e:
            raise RuntimeError(
                "Travel-style generation failed; refusing to substitute the input "
                "prompt as a generated style."
            ) from e

class TravelStyleRewardCalculator:
    """Compute similarity and rewards for travel-style descriptions."""

    def __init__(
        self,
        similarity_model: str = "../LLMs/Qwen3-Embedding-4B",
        device: str = "cuda:0",
    ):
        """
        Initialize the reward calculator.

        Args:
            similarity_model: Model used to compute text similarity.
        """
        self.similarity_model = SentenceTransformer(
            similarity_model,
            device=device,
        )
        logger.info(f"Reward calculator initialized with {similarity_model}")

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: Predicted travel-style text.
            text2: Ground-truth travel-style text.

        Returns:
            Similarity score in the range [0, 1].
        """
        try:
            # Encode texts with the similarity model
            embeddings = self.similarity_model.encode([text1, text2])

            # Compute cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

            # Clamp to [0, 1]
            similarity = max(0.0, min(1.0, similarity))

            return float(similarity)

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def calculate_reward(self, predicted_style: str, actual_style: str,
                        similarity_weight: float = 1.0, length_penalty: float = 0.1) -> float:
        """
        Compute the reinforcement learning reward.

        Args:
            predicted_style: Generated travel-style text.
            actual_style: Ground-truth travel-style text.
            similarity_weight: Weight for similarity reward.
            length_penalty: Weight for length-based penalty.

        Returns:
            Reward score.
        """
        # Base similarity reward
        similarity = self.calculate_similarity(predicted_style, actual_style)
        similarity_reward = similarity * similarity_weight

        # Length penalty to discourage overly long or short outputs
        predicted_length = len(predicted_style.split())
        actual_length = len(actual_style.split())
        length_diff = abs(predicted_length - actual_length) / max(actual_length, 1)
        length_penalty_score = max(0, 1 - length_diff) * length_penalty

        # Final reward
        total_reward = similarity_reward + length_penalty_score

        logger.debug(f"Similarity: {similarity:.3f}, Length penalty: {length_penalty_score:.3f}, Total reward: {total_reward:.3f}")

        return total_reward

    def get_embedding(self, texts: List[str], embedding_dim: Optional[int] = None) -> np.ndarray:
        """
        Compute embeddings for a batch of texts.

        Args:
            texts: List of input strings.
            embedding_dim: Desired embedding dimension; embeddings are truncated or padded accordingly.

        Returns:
            NumPy array of embeddings shaped (batch_size, embedding_dim).
        """
        try:
            # Encode texts with the similarity model
            embeddings = self.similarity_model.encode(texts)

            # Truncate or pad to the requested embedding dimension
            if embedding_dim is not None:
                if embedding_dim > embeddings.shape[1]:
                    logger.warning(f"Requested embedding_dim {embedding_dim} is larger than model output {embeddings.shape[1]}")
                    # Pad with zeros when the requested dimension exceeds the model output
                    padded_embeddings = np.zeros((embeddings.shape[0], embedding_dim))
                    padded_embeddings[:, :embeddings.shape[1]] = embeddings
                    embeddings = padded_embeddings
                else:
                    # Truncate to the specified dimension
                    embeddings = embeddings[:, :embedding_dim]

            logger.debug(f"Generated embeddings for {len(texts)} texts with shape {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return zero embeddings as a fallback
            fallback_dim = embedding_dim if embedding_dim is not None else 768  # Default dimension
            return np.zeros((len(texts), fallback_dim))

def travel_style_similarity_reward_func(similarity_model, prompts, completions, reference_responses, **kwargs) -> list[float]:
    """
    Reward function based on travel-style similarity.

    Args:
        similarity_model: Model instance used to compute similarity.
        prompts: List of input prompts.
        completions: List of model completions.
        reference_responses: List of reference responses.
        **kwargs: Additional keyword arguments.

    Returns:
        list[float]: Reward values for each completion.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    responses = [completion[0]['content'] if isinstance(completion, list) else completion for completion in completions]
    rewards = []

    print(f"\n{'='*80}")
    print(f"Similarity reward evaluation - processing {len(responses)} generations")
    print(f"{'='*80}")

    for i, response in enumerate(responses):
        if i < len(reference_responses):
            reference = reference_responses[i]

            try:
                # Encode texts
                embeddings = similarity_model.encode([response, reference])

                # Compute cosine similarity
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

                # Clamp to [0, 1]
                similarity = max(0.0, min(1.0, similarity))

                # Convert similarity into reward
                reward = similarity
                rewards.append(reward)

                # Detailed logging
                print(f"\nSample {i+1}:")
                print(f"Prompt: {prompts[i][:100]}..." if len(prompts[i]) > 100 else f"Prompt: {prompts[i]}")
                print(f"Generated response: {response[:150]}..." if len(response) > 150 else f"Generated response: {response}")
                print(f"Reference: {reference[:150]}..." if len(reference) > 150 else f"Reference: {reference}")
                print(f"Similarity: {similarity:.4f}")
                print(f"Similarity reward: {reward:.4f}")
                print(f"{'-'*60}")

            except Exception as e:
                logger.error(f"Error calculating similarity reward: {e}")
                rewards.append(0.0)
                print(f"Sample {i+1}: similarity computation failed, reward set to 0.0")
        else:
            rewards.append(0.0)
            print(f"Sample {i+1}: missing reference, reward set to 0.0")

    avg_similarity_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"\nAverage similarity reward: {avg_similarity_reward:.4f}")
    print(f"{'='*80}\n")

    return rewards

def travel_style_length_reward_func(prompts, completions, reference_responses, **kwargs) -> list[float]:
    """
    Reward function based on length consistency of travel-style outputs.

    Args:
        prompts: List of input prompts.
        completions: List of model completions.
        reference_responses: List of reference responses.
        **kwargs: Additional keyword arguments.

    Returns:
        list[float]: Reward values for each completion.
    """
    responses = [completion[0]['content'] if isinstance(completion, list) else completion for completion in completions]
    rewards = []

    print(f"\n{'='*80}")
    print(f"Length reward evaluation - processing {len(responses)} generations")
    print(f"{'='*80}")

    for i, response in enumerate(responses):
        if i < len(reference_responses):
            reference = reference_responses[i]

            predicted_length = len(response.split())
            reference_length = len(reference.split())

            # Compute relative length difference
            if reference_length > 0:
                length_diff = abs(predicted_length - reference_length) / reference_length
                # Higher reward when lengths align closely
                length_reward = max(0, 1 - length_diff) * 0.5
                rewards.append(length_reward)

                # Detailed logging
                print(f"Sample {i+1}:")
                print(f"Generated length: {predicted_length} words")
                print(f"Reference length: {reference_length} words")
                print(f"Length difference ratio: {length_diff:.4f}")
                print(f"Length reward: {length_reward:.4f}")
                print(f"{'-'*60}")
            else:
                rewards.append(0.0)
                print(f"Sample {i+1}: reference length is 0, reward set to 0.0")
        else:
            rewards.append(0.0)
            print(f"Sample {i+1}: missing reference, reward set to 0.0")

    avg_length_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"\nAverage length reward: {avg_length_reward:.4f}")
    print(f"{'='*80}\n")

    return rewards

class TravelStyleGRPOTrainer:
    """Train travel-style generators with the GRPO algorithm."""

    def __init__(self,
                 model_name: str = "../LLMs/Qwen3-8B",
                 similarity_model_name: str = "../LLMs/Qwen3-Embedding-4B",
                 device: str = "cuda",
                 use_lora: bool = True,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1,
                 use_accelerate: bool = False,
                 lora_config: str = None,
                 is_train: bool = True):
        """
        Initialize the GRPO trainer.

        Args:
            model_name: Name of the base language model.
            device: Target device for training.
            use_lora: Whether to fine-tune with LoRA adapters.
            lora_r: LoRA rank.
            lora_alpha: LoRA alpha parameter.
            lora_dropout: LoRA dropout rate.
            use_accelerate: Whether to use the accelerate library.
        """
        self.device = device
        self.model_name = model_name
        self.use_lora = use_lora
        self.use_accelerate = use_accelerate and ACCELERATE_AVAILABLE
        self.lora_config = lora_config
        
        # Initialize accelerator if requested
        if self.use_accelerate:
            self.accelerator = Accelerator()
        else:
            self.accelerator = None

        # Training progress tracking
        self.training_progress = None
        self.current_step = 0
        self.total_steps = 0

        if is_train:
            from sentence_transformers import SentenceTransformer

            # Initialize similarity model
            self.similarity_model = SentenceTransformer(similarity_model_name, device="cuda:2")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load base model
            if self.use_accelerate:
                # With accelerate, avoid setting device_map explicitly
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
            for param in self.model.parameters():
                param.requires_grad = False  # freeze the model - train adapters later


            # Configure LoRA adapters
            if self.use_lora:
                if self.lora_config is None:
                    self.lora_config = LoraConfig(
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        target_modules=["q_proj", "v_proj"],
                        lora_dropout=lora_dropout,
                        task_type="CAUSAL_LM"
                    )
                self.model = get_peft_model(self.model, self.lora_config)

        logger.info("GRPO trainer initialized with LoRA and Accelerate" if (use_lora and self.use_accelerate) 
                   else "GRPO trainer initialized with LoRA" if use_lora 
                   else "GRPO trainer initialized")

    def prepare_dataset(self, text_dataset=None, save_dataset=True, dataset_name = None, batch_size=4) -> Dict:
        """
        Prepare the training dataset.

        Args:
            text_dataset: TravelTextDataset instance containing prompts and references.
            save_dataset: Whether to persist the processed dataset locally.
            dataset_name: Dataset name used when saving to disk.
            batch_size: Batch size for batched LLM generation.

        Returns:
            A tuple of (datasets.Dataset, reference responses).
        """
        queries = []
        reference_responses = []

        # Ensure the output directory exists when saving
        dataset_dir = "../dataset"
        if save_dataset:
            os.makedirs(dataset_dir, exist_ok=True)
            print(f"Datasets will be saved to: {dataset_dir}")

        if text_dataset is not None:
            # Use the provided text dataset
            from data import TravelTextDataset

            print(f"\n{'='*80}")
            print(f"Starting text dataset processing - total samples: {len(text_dataset)}")
            print(f"Batch size: {batch_size} (multi-GPU friendly)")
            print(f"{'='*80}")

            # Generate reference answers via LLM
            destination_generator = TravelStyleGenerator(use_lora=self.use_lora, lora_path=self.lora_config if self.use_lora else None, use_vllm=False)

            # Collect items for batched processing
            all_items = []
            for item in text_dataset:
                all_items.append(item)

            # Prepare batched prompts
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

            print(f"\n🔄 Starting batched reference generation...")
            print(f"Total samples: {len(destination_prompts)}")
            print(f"Estimated batches: {(len(destination_prompts) + batch_size - 1) // batch_size}")

            # Batched reference generation
            batch_progress = tqdm(
                range(0, len(destination_prompts), batch_size),
                desc="Generating references in batches",
                unit="batch",
                total=(len(destination_prompts) + batch_size - 1) // batch_size
            )

            successful_count = 0
            failed_count = 0

            for batch_start in batch_progress:
                batch_end = min(batch_start + batch_size, len(destination_prompts))
                batch_destination_prompts = destination_prompts[batch_start:batch_end]
                batch_item_info = item_info[batch_start:batch_end]

                # Update progress details
                batch_progress.set_postfix({
                    'batch': f"{batch_start//batch_size + 1}/{(len(destination_prompts) + batch_size - 1) // batch_size}",
                    'batch_size': len(batch_destination_prompts),
                    'success': successful_count,
                    'failed': failed_count
                })


                # Generate references for the batch
                batch_references = destination_generator.get_output(batch_destination_prompts, max_length=512, temperature=0.7)

                # Process batch results
                for i, reference_style in enumerate(batch_references):
                    actual_index = batch_start + i
                    if reference_style != "Error in generation":
                        queries.append(hometown_prompts[actual_index])
                        reference_responses.append(reference_style)
                        successful_count += 1

                        # Detailed logging every 50 successful samples
                        if successful_count % 50 == 0:
                            current_info = batch_item_info[i]
                            print(f"\n📊 Processed {successful_count} samples successfully")
                            print(f"   Current sample - user: {current_info['uid']}, destination: {current_info['dst_region']}")
                            print(f"   Reference length: {len(reference_style)} characters")
                            print(f"   Reference preview: {reference_style[:100]}...")
                            print(f"{'-'*60}")
                    else:
                        failed_count += 1
                        current_info = batch_item_info[i]
                        logger.warning(f"Failed to generate reference for user {current_info['uid']}")

                # Update aggregate progress statistics
                batch_progress.set_postfix({
                    'batch': f"{batch_start//batch_size + 1}/{(len(destination_prompts) + batch_size - 1) // batch_size}",
                    'success': successful_count,
                    'failed': failed_count,
                    'success_rate': f"{(successful_count/(successful_count+failed_count))*100:.1f}%" if (successful_count+failed_count) > 0 else "0%"
                })

            batch_progress.close()

            print(f"\n✅ Finished processing text dataset!")
            print(f"   Total samples: {len(text_dataset)}")
            print(f"   Successful generations: {successful_count}")
            print(f"   Failures: {failed_count}")
            print(f"   Success rate: {(successful_count/(successful_count+failed_count))*100:.1f}%")
            print(f"   Batch throughput: {successful_count/((len(destination_prompts) + batch_size - 1) // batch_size):.1f} samples per batch on average")
        else:
            raise ValueError("Either text_dataset or trajectories must be provided")

        # Build Hugging Face dataset object
        from datasets import Dataset
        dataset = Dataset.from_dict({
            "prompt": queries,
            "reference": reference_responses
        })

        print(f"\n📦 Dataset creation complete!")
        print(f"   Final dataset size: {len(dataset)}")
        print(f"   Number of prompts: {len(queries)}")
        print(f"   Number of references: {len(reference_responses)}")

        # Optionally persist the dataset
        if save_dataset and len(dataset) > 0:
            try:
                print(f"\n💾 Saving dataset to disk...")

                # Timestamped folder name
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                dataset_name = f"{dataset_name}_grpo_{timestamp}"
                dataset_path = os.path.join(dataset_dir, dataset_name)

                # Persist dataset to disk
                dataset.save_to_disk(dataset_path)

                # Persist additional metadata
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

                print(f"✅ Dataset saved successfully!")
                print(f"   Output path: {dataset_path}")
                print(f"   Metadata file: {metadata_path}")

                # Save preview samples
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

                print(f"   Sample preview: {sample_path}")

            except Exception as e:
                logger.error(f"Failed to save dataset: {e}")
                print(f"❌ Failed to save dataset: {e}")

        print(f"\n{'='*80}")
        print(f"Dataset preparation complete!")
        print(f"Batch mode enabled, batch size: {batch_size}")
        print(f"{'='*80}\n")

        return dataset, reference_responses

    def train(self, text_dataset=None,
              output_dir: str = "./grpo_travel_style_model",
              run_name: str = "travel_style_grpo",
              num_train_epochs: int = 2,
              learning_rate: float = 5e-5,
              per_device_train_batch_size: int = 2,
              gradient_accumulation_steps: int = 1,
              num_generations: int = 4,
              max_prompt_length: int = 2048,
              max_completion_length: int = 256,
              save_steps: int = 250,
              logging_steps: int = 1,
              is_gspo: bool = False):
        """
        Train the model with the GRPO algorithm.

        Args:
            text_dataset: Hugging Face dataset containing prompts and references.
            output_dir: Directory where checkpoints should be saved.
            run_name: Name for the training run.
            num_train_epochs: Number of training epochs.
            learning_rate: Optimizer learning rate.
            per_device_train_batch_size: Batch size per device.
            gradient_accumulation_steps: Gradient accumulation steps.
            num_generations: Number of generations per prompt.
            max_prompt_length: Maximum prompt length.
            max_completion_length: Maximum completion length.
            save_steps: Step interval for checkpointing.
            logging_steps: Step interval for logging.
            is_gspo: Whether to use GSPO configuration variants.
        """
        print(f"\n{'='*100}")
        print(f"Starting GRPO reinforcement learning")
        print(f"{'='*100}")

        # Prepare dataset
        print("Preparing training dataset...")
        dataset = text_dataset

        # Compute total training steps
        dataset_size = len(dataset)
        self.total_steps = (dataset_size * num_train_epochs) // (per_device_train_batch_size * gradient_accumulation_steps)

        print(f"Dataset size: {dataset_size}")
        print(f"Epochs: {num_train_epochs}")
        print(f"Batch size per device: {per_device_train_batch_size}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Estimated total steps: {self.total_steps}")
        print(f"Output directory: {output_dir}")

        # Initialize progress bar
        self.training_progress = tqdm(
            total=self.total_steps,
            desc="GRPO training progress",
            unit="step",
            position=0,
            leave=True
        )

        # Initialize SwanLab experiment tracking
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
            if is_gspo:
                swanlab.init(
                    project="travel-style-gspo",
                    experiment_name=run_name,
                    config=config_dict
                )
            else:
                swanlab.init(
                    project="travel-style-grpo",
                    experiment_name=run_name,
                    config=config_dict
                )
            logger.info("SwanLab experiment initialized")
        else:
            logger.warning("SwanLab not available, metrics will not be logged")

        # Configure GRPO/GSPO arguments
        if is_gspo:
            training_args = GRPOConfig(
                output_dir=output_dir,
                importance_sampling_level="sequence",
                loss_type="grpo",
                beta=0.0,
                # GSPO set KL regularization to zero: https://github.com/volcengine/verl/pull/2775#issuecomment-3131807306
                epsilon=3e-4,  # GSPO paper (v2), section 5.1
                epsilon_high=4e-4,  # GSPO paper (v2), section 5.1
                gradient_accumulation_steps=1,
                steps_per_generation=4,
                # partition rollout batch into 4 mini-batches. GSPO paper (v2), section 5.1. Must be 4 times gradient_accumulation_steps
                # Disable WandB integration
                report_to=None,  # Do not report to external services
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
                num_generations=num_generations,
                max_prompt_length=max_prompt_length,
                max_completion_length=max_completion_length,
                num_train_epochs=num_train_epochs,
                save_steps=save_steps,
                max_grad_norm=0.1,
                log_on_each_node=False,
                use_vllm=False,
            )
        else:
            training_args = GRPOConfig(
                output_dir=output_dir,
                # Disable WandB integration
                report_to=None,
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

    # Define reward functions that leverage closure over reference responses and progress tracking
        def similarity_reward_func(prompts, completions, reference, **kwargs):
            # Update progress bar
            self.current_step += 1
            self.training_progress.update(1)
            self.training_progress.set_postfix({
                'Step': f"{self.current_step}/{self.total_steps}",
                'Phase': 'Similarity reward computation'
            })

            print(f"\n🔄 Step {self.current_step}/{self.total_steps}: similarity reward computation")
            similarity_rewards = travel_style_similarity_reward_func(self.similarity_model, prompts, completions, reference, **kwargs)
            # Log metrics to SwanLab
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
                'Phase': 'Length reward computation'
            })

            print(f"\n📏 Step {self.current_step}/{self.total_steps}: length reward computation")
            return travel_style_length_reward_func(prompts, completions, reference, **kwargs)

        def quality_reward_func(prompts, completions, reference, **kwargs):
            self.training_progress.set_postfix({
                'Step': f"{self.current_step}/{self.total_steps}",
                'Phase': 'Quality reward computation'
            })

            # Combine rewards and display summary
            similarity_rewards = travel_style_similarity_reward_func(self.similarity_model, prompts, completions, reference, **kwargs)
            length_rewards = travel_style_length_reward_func(prompts, completions, reference, **kwargs)

            total_rewards = [s + l for s, l in zip(similarity_rewards, length_rewards)]
            avg_total_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0

            print(f"\n🎯 Step {self.current_step}/{self.total_steps} summary:")
            print(f"   Average total reward: {avg_total_reward:.4f}")
            print(f"   Progress: {(self.current_step/self.total_steps)*100:.1f}%")

            # Log metrics to SwanLab
            if SWANLAB_AVAILABLE:
                swanlab.log({
                    "step": self.current_step,
                    "reward/similarity": sum(similarity_rewards) / len(similarity_rewards) if similarity_rewards else 0.0,
                    "reward/length": sum(length_rewards) / len(length_rewards) if length_rewards else 0.0,
                    "reward/total": avg_total_reward,
                    "progress": (self.current_step/self.total_steps)*100
                })

            return total_rewards

        # Instantiate the GRPO trainer
        if self.use_accelerate:
            # Prepare model and dataset with accelerate
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
            trainer.current_gradient_accumulation_steps = 1  # Align with grpo_trainer.py expectations

        logger.info("Starting GRPO training...")
        print(f"\n🚀 Launching reinforcement learning...")
        print(f"Model: {self.model_name}")
        print(f"LoRA enabled: {'yes' if self.use_lora else 'no'}")
        if self.use_accelerate:
            print(f"Accelerate enabled: yes")

        # Record training start time
        start_time = time.time()
        if SWANLAB_AVAILABLE:
            swanlab.log({"training_status": "started", "start_time": start_time})

        try:
            trainer.train()
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Ensure progress bar is closed
            if self.training_progress:
                self.training_progress.close()

        # Compute training duration
        end_time = time.time()
        training_duration = end_time - start_time

        print(f"\n✅ Training complete!")
        print(f"Elapsed time: {training_duration:.2f} seconds ({training_duration/60:.1f} minutes)")
        print(f"Total steps: {self.current_step}")

        # Log completion to SwanLab
        if SWANLAB_AVAILABLE:
            swanlab.log({
                "training_status": "completed",
                "end_time": end_time,
                "training_duration": training_duration,
                "final_step": self.current_step
            })

        # Save the trained model
        print(f"\n💾 Saving model to: {output_dir}")
        trainer.save_model(output_dir)
        logger.info(f"Model saved to {output_dir}")

        # Record model-saving metadata
        if SWANLAB_AVAILABLE:
            swanlab.log({
                "model_saved": True,
                "model_path": output_dir,
                "final_epoch": num_train_epochs
            })

        print(f"\n{'='*100}")
        print(f"GRPO reinforcement learning finished!")
        print(f"{'='*100}\n")

def main():
    """Entry point demonstrating trainer usage."""
    import wandb
    import argparse
    wandb.init(mode="disabled")  # Force-disable wandb
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_accelerate", action="store_true", help="Enable accelerate for training")
    args = parser.parse_args()
    # Initialize trainer (LoRA + optional accelerate)
    trainer = TravelStyleGRPOTrainer(
        model_name="../LLMs/Qwen3-8B",
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        use_accelerate=args.use_accelerate
    )

    # Load training dataset
    from datasets import load_from_disk
    text_dataset = load_from_disk("../dataset/Yelp_20250714_192438")

    # Train the model
    trainer.train(
        text_dataset=text_dataset,
        output_dir="./grpo_Yelp_lora_model",
        run_name="travel_Yelp_style_grpo_lora",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        is_gspo=False
    )

    # Evaluation placeholder
    # results = trainer.evaluate(test_trajectories)
    # print(f"Evaluation results: {results}")
    
    # Finalize SwanLab tracking
    if SWANLAB_AVAILABLE:
        swanlab.finish()
        print("SwanLab experiment finished")

if __name__ == "__main__":
    main()
