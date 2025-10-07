# -*- coding: UTF-8 -*-
"""
LLMs.py - Interface for large language models and reinforcement learning training module.

This module provides:
1. Interfaces for invoking a large language model to generate travel-style responses.
2. Reinforcement learning training interfaces for the LLM using the GRPO method.
3. Reward functions for text similarity and accuracy.
4. F1-score reward: treat RL-generated text as the target LLM input to the saved SPOTModel,
    run inference on the corresponding sample to obtain the predicted trajectory, and compute
    the sample_f1 against the ground-truth trajectory as the reward.
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'  # Specify visible GPU devices
import sys

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments
)
from transformers.trainer_callback import TrainerCallback
from trl import GRPOConfig, GRPOTrainer, SFTTrainer
from peft import LoraConfig, get_peft_model
from data import TravelDataset
from tqdm import tqdm
import time
import logging
import pickle
from metrics import category_consistency_rate, hit_rate, recall_rate, diversity_rate
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


# ===== Dependencies required for loading the SPOT-Trip model and data for the F1 reward =====
from data import TravelDataset, random_split  # Dataset utilities
from utils import collate_fn  # Collate function for the DataLoader
from model import SemPOIModel  # Main model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Standalone reward functions (following the pattern in test.py)
def travel_style_similarity_reward_func(similarity_model, prompts, completions, reference_responses, **kwargs) -> list[
    float]:
    """
    Travel-style similarity reward function.

    Args:
        similarity_model: Model instance used to compute similarity.
        prompts: List of input prompts.
        completions: List of model-generated completions.
        reference_responses: List of reference answers.
        **kwargs: Additional optional arguments.

    Returns:
        list[float]: List of reward values.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    responses = [completion[0]['content'] if isinstance(completion, list) else completion for completion in completions]
    rewards = []

    print(f"\n{'=' * 80}")
    print(f"Similarity reward computation - processing {len(responses)} generated responses")
    print(f"{'=' * 80}")

    for i, response in enumerate(responses):
        if i < len(reference_responses):
            reference = reference_responses[i]

            try:
                # Encode text
                embeddings = similarity_model.encode([response, reference])

                # Compute cosine similarity
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

                # Ensure similarity stays within [0, 1]
                similarity = max(0.0, min(1.0, similarity))

                # Convert similarity to reward score
                reward = similarity
                rewards.append(reward)

                # Log detailed information
                print(f"\nSample {i + 1}:")
                print(f"Prompt: {prompts[i][:100]}..." if len(prompts[i]) > 100 else f"Prompt: {prompts[i]}")
                print(f"Generated reply: {response[:150]}..." if len(response) > 150 else f"Generated reply: {response}")
                print(f"Reference answer: {reference[:150]}..." if len(reference) > 150 else f"Reference answer: {reference}")
                print(f"Similarity: {similarity:.4f}")
                print(f"Similarity reward: {reward:.4f}")
                print(f"{'-' * 60}")

            except Exception as e:
                logger.error(f"Error calculating similarity reward: {e}")
                rewards.append(0.0)
                print(f"Sample {i + 1}: Similarity computation failed, reward set to 0.0")
        else:
            rewards.append(0.0)
            print(f"Sample {i + 1}: Missing reference answer, reward set to 0.0")

    avg_similarity_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"\nAverage similarity reward: {avg_similarity_reward:.4f}")
    print(f"{'=' * 80}\n")

    return rewards


def travel_style_length_reward_func(prompts, completions, reference_responses, **kwargs) -> list[float]:
    """
    Travel-style length reward function.

    Args:
        prompts: List of input prompts.
        completions: List of model-generated completions.
        reference_responses: List of reference answers.
        **kwargs: Additional optional arguments.

    Returns:
        list[float]: List of reward values.
    """
    responses = [completion[0]['content'] if isinstance(completion, list) else completion for completion in completions]
    rewards = []

    print(f"\n{'=' * 80}")
    print(f"Length reward computation - processing {len(responses)} generated responses")
    print(f"{'=' * 80}")

    for i, response in enumerate(responses):
        if i < len(reference_responses):
            reference = reference_responses[i]

            predicted_length = len(response.split())
            reference_length = len(reference.split())

            # Compute the length difference ratio
            if reference_length > 0:
                length_diff = abs(predicted_length - reference_length) / reference_length
                # The closer the lengths, the higher the reward
                length_reward = max(0, 1 - length_diff) * 0.5
                rewards.append(length_reward)

                # Log detailed information
                print(f"Sample {i + 1}:")
                print(f"Generated length: {predicted_length} words")
                print(f"Reference length: {reference_length} words")
                print(f"Length difference ratio: {length_diff:.4f}")
                print(f"Length reward: {length_reward:.4f}")
                print(f"{'-' * 60}")
            else:
                rewards.append(0.0)
                print(f"Sample {i + 1}: Reference length is 0, reward set to 0.0")
        else:
            rewards.append(0.0)
            print(f"Sample {i + 1}: Missing reference answer, reward set to 0.0")

    avg_length_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"\nAverage length reward: {avg_length_reward:.4f}")
    print(f"{'=' * 80}\n")

    return rewards


# ===================== F1 reward based on the SPOTModel ===================== #
class _RLArgsStub:
    """Minimal parameter object that satisfies the fields required for SPOTModel inference."""
    # Keep defaults consistent with main.py; override in TravelStyleGRPOTrainer as needed
    def __init__(self, device="cuda:1", use_llm=True, use_target_llm=True,
                 use_vllm=False, use_lora=False, lora_path="./grpo_travel_style_lora_model/checkpoint-5500",
                 llm_embedding_dim=256, hidden_size=256,
                 kg=False, ode=False, s_infer=False, st_module=True,
                 lm_hid_layers=3, lm_latent_dim=128,
                 dyn_hid_layers=3, dyn_latent_dim=128,
                 tau=0.2, sig_v=0.6, confidence=0.5,
                 num_semantic_parts=8, lambda_diversity=0.1,
                 lambda_attn_reg=0.1, mask_ratio=0.75,
                 dataset_name: str = "Yelp"):
        self.device = device
        self.use_llm = use_llm
        self.use_target_llm = use_target_llm
        self.use_vllm = use_vllm
        self.use_lora = use_lora
        self.lora_path = lora_path
        self.llm_embedding_dim = llm_embedding_dim
        self.hidden_size = hidden_size
        self.kg = kg
        self.ode = ode
        self.s_infer = s_infer
        self.st_module = st_module
        self.lm_hid_layers = lm_hid_layers
        self.lm_latent_dim = lm_latent_dim
        self.dyn_hid_layers = dyn_hid_layers
        self.dyn_latent_dim = dyn_latent_dim
        self.tau = tau
        self.sig_v = sig_v
        self.confidence = confidence
        self.dataset_name = dataset_name
        self.num_semantic_parts = num_semantic_parts
        self.lambda_diversity = lambda_diversity
        self.lambda_attn_reg = lambda_attn_reg
        self.mask_ratio = mask_ratio


class F1RewardEvaluator:
    """
    Lazily load the SPOTModel checkpoint and dataset, feed RL-generated text as messages into the
    model, compute the sample-level F1 (sample_f1) against the ground-truth destination trajectory,
    and return it as the reward.
    """
    def __init__(self, dataset_name: str = "Yelp", run_name: str = "default",
                 device: str = "cuda:1"):
        self.dataset_name = dataset_name
        self.run_name = run_name
        self.device = device
        self.poi_meta = None
        try:
            with open(f'../{dataset_name}/poi_meta.pkl', 'rb') as f:
                self.poi_meta = pickle.load(f)
        except Exception as _e:
            logger.log(f"[warn] Failed to load poi_meta.pkl: {_e}")

        self._data = None
        self._model = None
        self._offset = 0  # Pointer aligned with the RL batch
        self._dl_cache = {}
        # Load immediately (no lazy loading)
        self._lazy_load_data_and_model()

    # ---------- Internal utilities ---------- #
    def _find_checkpoint(self) -> str:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../{self.dataset_name}/model_save/{self.run_name}"))
        if not os.path.isdir(base):
            raise FileNotFoundError(f"Model directory not found: {base}")
        if self.dataset_name == "Foursquare":
            best = os.path.join(base, "model_0.xhr")
        else:
            best = os.path.join(base, "model_5.xhr")
        if os.path.exists(best):
            return best
        else:
            raise FileNotFoundError(f"Checkpoint not found in: {base}")


    def _lazy_load_data_and_model(self):
        # Already invoked by the constructor; return immediately if everything is loaded
        if self._data is not None and self._model is not None:
            return

        # Load the raw trajectory data (following the same path convention as in main.py)
        base_dir = os.path.dirname(__file__)
        ori_path = os.path.abspath(os.path.join(base_dir, f"../{self.dataset_name}/home.txt"))
        dst_path = os.path.abspath(os.path.join(base_dir, f"../{self.dataset_name}/oot.txt"))
        trans_path = os.path.abspath(os.path.join(base_dir, f"../{self.dataset_name}/travel.txt"))

        args_stub = _RLArgsStub(
            device=self.device,
            use_llm=True,
            use_target_llm=True,
            dataset_name=self.dataset_name,
        )

        self._data = TravelDataset(args_stub, ori_path, dst_path, trans_path)

    # Build the model (following main.py)
        max_d_length = max(len(seq) for seq in self._data.dsts)
        max_o_length = max(len(seq) for seq in self._data.oris)

        model = SemPOIModel(
            args_stub,
            poi_size=len(self._data.poi_idx) + 1,
            region_poi=self._data.region_poi,
            max_length_venue_id=max_d_length,
            max_length_ori_id=max_o_length,
            d_model=args_stub.hidden_size,
            n_head=4,
            num_encoder_layers=1,
            d_z=args_stub.hidden_size,
            kg_dataset=None,
        )
        model = model.to(self.device)

    # Load model weights
        ckpt = torch.load(self._find_checkpoint(), map_location=self.device)
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        self._model = model

    @staticmethod
    def _tensorize_messages(completions) -> list[str]:
        msgs = []
        for c in completions:
            if isinstance(c, list):
                # Support OpenAI/TRL format: [{"content": "..."}, ...]
                if len(c) > 0 and isinstance(c[0], dict) and "content" in c[0]:
                    msgs.append(c[0]["content"])
                else:
                    msgs.append(str(c))
            elif isinstance(c, dict) and "content" in c:
                msgs.append(c["content"])
            else:
                msgs.append(str(c))
        return msgs

    @staticmethod
    def _sample_f1(pred_ids: torch.Tensor, true_ids: torch.Tensor) -> float:
        # Remove padding token 0
        p = [int(x) for x in pred_ids.tolist() if int(x) != 0]
        t = [int(x) for x in true_ids.tolist() if int(x) != 0]
        if len(p) == 0 and len(t) == 0:
            return 1.0
        if len(p) == 0 or len(t) == 0:
            return 0.0
        ps, ts = set(p), set(t)
        inter = len(ps & ts)
        denom = len(ps) + len(ts)
        if denom == 0:
            return 0.0
        return 2.0 * inter / denom

    def _hit_num(self, pred_ids: torch.Tensor, true_ids: torch.Tensor) -> int:
        # Remove padding token 0
        p = [int(x) for x in pred_ids.tolist() if int(x) != 0]
        t = [int(x) for x in true_ids.tolist() if int(x) != 0]
        if len(p) == 0 and len(t) == 0:
            return 0.0
        if len(p) == 0 or len(t) == 0:
            return 0.0
        ps, ts = set(p), set(t)
        inter = len(ps & ts)
        return inter

    def _get_subset_loader(self, indices: list[int], batch_size: int):
        """
        Return a DataLoader whose ordering and batch boundaries match the original loader,
        based on the provided sample indices:
        - Use Subset to preserve the relative order of the provided indices (no need to shuffle).
        - Reuse the same collate_fn.
        - Let the caller provide batch_size to replicate the original batch splits.

        Note: PyTorch's Subset indexes according to the order of the provided indices; with
        shuffle=False the DataLoader iterates over the Subset sequentially, ensuring batch
        alignment.
        """
        from torch.utils.data import Subset, DataLoader

        if not indices:
            # Empty indices return an empty DataLoader (no batches will be produced)
            return DataLoader([], batch_size=1)

        subset = Subset(self._data, indices)
        # Disable shuffling to iterate strictly in the order of indices; reuse the original collate_fn
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        return loader

    def compute_batch_f1(self, prompts, completions) -> list[float]:
        """
        Given a batch of RL-generated text, pick the samples in the dataset with matching order,
        feed the text into the model as messages, and return the sample_f1 for each sample.
        """
        self._lazy_load_data_and_model()
        msgs = self._tensorize_messages(completions)

        bsz = len(msgs)
        N = len(self._data)
        # Take consecutive indices with wrap-around if necessary
        idxs = [ (self._offset + i) % N for i in range(bsz) ]
        self._offset = (self._offset + bsz) % N

        loader = self._get_subset_loader(idxs, batch_size=bsz)

        f1_scores: list[float] = []
        # Iterate through a batch (ideally only one because batch_size == bsz)
        with torch.no_grad():
            for (uid, o_ck, d_ck, masked_d_ck, o_h, d_h, masked_d_h, o_t, d_t, o_l, d_l, o_pad, d_pad, o_rg, d_rg) in loader:
                # Feed the messages list to the forward pass
                uid = uid.to(self.device)
                o_ck = o_ck.to(self.device)
                masked_d_ck = masked_d_ck.to(self.device)
                d_ck = d_ck.to(self.device)
                o_h = o_h.to(self.device)
                masked_d_h = masked_d_h.to(self.device)
                d_h = d_h.to(self.device)
                o_t = o_t.to(self.device)
                d_t = d_t.to(self.device)
                o_l = o_l.to(self.device)
                d_l = d_l.to(self.device)
                o_pad = o_pad.to(self.device)
                d_pad = d_pad.to(self.device)
                o_rg = o_rg.to(self.device)
                d_rg = d_rg.to(self.device)

                # Predict
                predicted_ids = self._model(uid, msgs, o_ck, masked_d_ck, o_t, d_t, o_l, d_l, o_pad, d_pad, d_ck, o_rg,
                                      d_rg, target_seq=None)

                # Compute F1 per sample
                for i in range(predicted_ids.shape[0]):
                    sample_pred = predicted_ids[i].cpu()  # shape: [seq_len]
                    sample_target = d_ck[i].cpu()  # shape: [seq_len]

                    # Exclude padded values (assuming padding is represented by 0)
                    non_padded_indices = sample_target != 0
                    sample_pred = sample_pred[non_padded_indices]
                    sample_target = sample_target[non_padded_indices]
                    sample_pred = sample_pred[1:-1]  # Exclude start and end tokens
                    sample_target = sample_target[1:-1]  # Exclude start and end tokens
                    f1 = self._sample_f1(sample_pred, sample_target)
                    f1_scores.append(float(max(0.0, min(1.0, f1))))

        # Align with the number of inputs (should match theoretically)
        if len(f1_scores) < bsz:
            f1_scores += [0.0] * (bsz - len(f1_scores))
        elif len(f1_scores) > bsz:
            f1_scores = f1_scores[:bsz]
        return f1_scores

    def compute_batch_f1_category(self, prompts, completions) -> list[float]:
        """
        Given a batch of RL-generated text, pick the samples in the dataset with matching order,
        feed the text into the model as messages, and return sample_f1 and category consistency
        for each sample.
        """
        self._lazy_load_data_and_model()
        msgs = self._tensorize_messages(completions)

        bsz = len(msgs)
        N = len(self._data)
        # Take consecutive indices with wrap-around if necessary
        idxs = [ (self._offset + i) % N for i in range(bsz) ]
        self._offset = (self._offset + bsz) % N

        loader = self._get_subset_loader(idxs, batch_size=bsz)

        f1_scores: list[float] = []
        cat_scores: list[float] = []
        # Iterate through a batch (ideally only one because batch_size == bsz)
        with torch.no_grad():
            for (uid, o_ck, d_ck, masked_d_ck, o_h, d_h, masked_d_h, o_t, d_t, o_l, d_l, o_pad, d_pad, o_rg, d_rg) in loader:
                # Feed the messages list to the forward pass
                uid = uid.to(self.device)
                o_ck = o_ck.to(self.device)
                masked_d_ck = masked_d_ck.to(self.device)
                d_ck = d_ck.to(self.device)
                o_h = o_h.to(self.device)
                masked_d_h = masked_d_h.to(self.device)
                d_h = d_h.to(self.device)
                o_t = o_t.to(self.device)
                d_t = d_t.to(self.device)
                o_l = o_l.to(self.device)
                d_l = d_l.to(self.device)
                o_pad = o_pad.to(self.device)
                d_pad = d_pad.to(self.device)
                o_rg = o_rg.to(self.device)
                d_rg = d_rg.to(self.device)

                # Predict
                predicted_ids = self._model(uid, msgs, o_ck, masked_d_ck, o_t, d_t, o_l, d_l, o_pad, d_pad, d_ck, o_rg,
                                      d_rg, target_seq=None)

                # Compute F1 per sample
                for i in range(predicted_ids.shape[0]):
                    sample_pred = predicted_ids[i].cpu()  # shape: [seq_len]
                    sample_target = d_ck[i].cpu()  # shape: [seq_len]

                    # Exclude padded values (assuming padding is represented by 0)
                    non_padded_indices = sample_target != 0
                    sample_pred = sample_pred[non_padded_indices]
                    sample_target = sample_target[non_padded_indices]
                    sample_pred = sample_pred[1:-1]  # Exclude start and end tokens
                    sample_target = sample_target[1:-1]  # Exclude start and end tokens
                    f1 = self._sample_f1(sample_pred, sample_target)
                    f1_scores.append(f1)
                    # Compute category consistency
                    cat_rate = category_consistency_rate(sample_pred, sample_target, self.poi_meta)
                    cat_scores.append(cat_rate)

        # Align with the number of inputs (should match theoretically)
        if len(f1_scores) < bsz:
            f1_scores += [0.0] * (bsz - len(f1_scores))
        elif len(f1_scores) > bsz:
            f1_scores = f1_scores[:bsz]
        return f1_scores, cat_scores

    def compute_Refine_POI_reward(self, prompts, completions) -> list[float]:
        """
        Given a batch of RL-generated text, pick the samples in the dataset with matching order,
        feed the text into the model as messages, and return hit, recall, diversity, and category
        consistency rewards per sample.
        """
        self._lazy_load_data_and_model()
        msgs = self._tensorize_messages(completions)

        bsz = len(msgs)
        N = len(self._data)
        # Take consecutive indices with wrap-around if necessary
        idxs = [ (self._offset + i) % N for i in range(bsz) ]
        self._offset = (self._offset + bsz) % N

        loader = self._get_subset_loader(idxs, batch_size=bsz)

        hit_scores: list[float] = []
        recall_scores: list[float] = []
        devisity_scores: list[float] = []
        cat_scores: list[float] = []
        # Iterate through a batch (ideally only one because batch_size == bsz)
        with torch.no_grad():
            for (uid, o_ck, d_ck, masked_d_ck, o_h, d_h, masked_d_h, o_t, d_t, o_l, d_l, o_pad, d_pad, o_rg, d_rg) in loader:
                # Feed the messages list to the forward pass
                uid = uid.to(self.device)
                o_ck = o_ck.to(self.device)
                masked_d_ck = masked_d_ck.to(self.device)
                d_ck = d_ck.to(self.device)
                o_h = o_h.to(self.device)
                masked_d_h = masked_d_h.to(self.device)
                d_h = d_h.to(self.device)
                o_t = o_t.to(self.device)
                d_t = d_t.to(self.device)
                o_l = o_l.to(self.device)
                d_l = d_l.to(self.device)
                o_pad = o_pad.to(self.device)
                d_pad = d_pad.to(self.device)
                o_rg = o_rg.to(self.device)
                d_rg = d_rg.to(self.device)

                # Predict
                predicted_ids = self._model(uid, msgs, o_ck, masked_d_ck, o_t, d_t, o_l, d_l, o_pad, d_pad, d_ck, o_rg,
                                      d_rg, target_seq=None)

                # Compute rewards per sample
                for i in range(predicted_ids.shape[0]):
                    sample_pred = predicted_ids[i].cpu()  # shape: [seq_len]
                    sample_target = d_ck[i].cpu()  # shape: [seq_len]

                    # Exclude padded values (assuming padding is represented by 0)
                    non_padded_indices = sample_target != 0
                    sample_pred = sample_pred[non_padded_indices]
                    sample_target = sample_target[non_padded_indices]
                    sample_pred = sample_pred[1:-1]  # Exclude start and end tokens
                    sample_target = sample_target[1:-1]  # Exclude start and end tokens
                    hit = hit_rate(sample_pred, sample_target)
                    hit_scores.append(hit)
                    # Compute category consistency
                    cat_rate = category_consistency_rate(sample_pred, sample_target, self.poi_meta)
                    cat_scores.append(cat_rate)
                    recall = recall_rate(sample_pred, sample_target)
                    recall_scores.append(recall)
                    diversity = diversity_rate(sample_pred)
                    devisity_scores.append(diversity)



        return hit_scores, recall_scores, devisity_scores, cat_scores

def travel_style_f1_reward_func(evaluator: F1RewardEvaluator, prompts, completions, reference_responses, **kwargs) -> list[float]:
    """
    Compute the sample-level F1 reward using the saved SPOTModel.
    The return value is within [0, 1], matching the semantics of sample_f1 in trainer.py.
    """
    # Directly call the evaluator
    # try:
    #     rewards = evaluator.compute_batch_f1(prompts, completions)
    # except Exception as e:
    #     print(f"[F1-Reward] computation failed: {e}")
    #     rewards = [0.0 for _ in range(len(completions))]
    rewards = evaluator.compute_batch_f1(prompts, completions)
    # Print summary
    if rewards:
        print(f"F1 reward (mean): {sum(rewards)/len(rewards):.4f} | Samples: {len(rewards)}")
    return rewards

def travel_style_f1_category_reward_func(evaluator: F1RewardEvaluator, prompts, completions, reference_responses, **kwargs) -> list[float]:
    """
    Compute the sample-level F1 reward using the saved SPOTModel.
    The return value is within [0, 1], matching the semantics of sample_f1 in trainer.py.
    """
    # Directly call the evaluator
    # try:
    #     rewards = evaluator.compute_batch_f1(prompts, completions)
    # except Exception as e:
    #     print(f"[F1-Reward] computation failed: {e}")
    #     rewards = [0.0 for _ in range(len(completions))]
    f1_rewards, cat_rewards = evaluator.compute_batch_f1_category(prompts, completions)
    # Print summary
    if f1_rewards:
        print(f"F1 reward (mean): {sum(f1_rewards)/len(f1_rewards):.4f} | Samples: {len(f1_rewards)}")
    if cat_rewards:
        print(f"Category consistency reward (mean): {sum(cat_rewards)/len(cat_rewards):.4f} | Samples: {len(cat_rewards)}")
    return f1_rewards, cat_rewards

class TravelStyleGRPOTrainer:
    """Train a travel-style generation model using the GRPO method."""

    def __init__(self,
                 model_name: str = "../LLMs/Qwen3-8B",
                 similarity_model_name: str = "../LLMs/Qwen3-Embedding-4B",
                 device: str = "cuda:0",
                 use_lora: bool = True,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1,
                 use_accelerate: bool = False,
                 lora_config: str = None,
                 is_train: bool = True,
                 dataset_name: str = "Yelp",
                 model_run_name: str = "default",
                 is_sft: bool = False,
                 need_similarity_model: bool = True):
        """
        Initialize the GRPO trainer.

        Args:
            model_name: Name of the base model.
            device: Compute device.
            use_lora: Whether to fine-tune with LoRA.
            lora_r: LoRA rank parameter.
            lora_alpha: LoRA alpha parameter.
            lora_dropout: LoRA dropout rate.
            use_accelerate: Whether to leverage the accelerate library.
        """
        self.device = device
        self.model_name = model_name
        self.use_lora = use_lora
        self.use_accelerate = use_accelerate and ACCELERATE_AVAILABLE
        self.lora_config = lora_config
        # SPOT-Trip configuration used for the F1 reward
        self.dataset_name = dataset_name
        self.model_run_name = model_run_name
        self.need_similarity_model = need_similarity_model

        # Training progress tracking
        self.training_progress = None
        self.current_step = 0
        self.total_steps = 0

        # F1 reward evaluator: pre-initialize on cuda:1
        if not is_sft:
            try:
                self.f1_evaluator = F1RewardEvaluator(
                    dataset_name=self.dataset_name,
                    run_name=self.model_run_name,
                    device="cuda:1",
                )
            except Exception as _e:
                logger.error(f"Failed to pre-load F1RewardEvaluator: {_e}")
                self.f1_evaluator = None

        if is_train:
            # Initialize similarity model
            if not is_sft and need_similarity_model:
                from sentence_transformers import SentenceTransformer
                self.similarity_model = SentenceTransformer(similarity_model_name, device="cuda:3")

            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load the model
            if self.use_accelerate:
                # When using accelerate, do not set device_map
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    # max_memory={0: "20GiB", 1: "20GiB"},
                    torch_dtype=torch.bfloat16
                )
            for param in self.model.parameters():
                param.requires_grad = False  # freeze the model - train adapters later

            # Configure LoRA
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
                else:
                    from peft import PeftModel
                    self.model = PeftModel.from_pretrained(self.model, self.lora_config, is_trainable=False)
                    # Add a new LoRA adapter
                    self.lora_config = LoraConfig(
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        target_modules=["q_proj", "v_proj"],
                        lora_dropout=lora_dropout,
                        task_type="CAUSAL_LM"
                    )
                    self.model = get_peft_model(self.model, self.lora_config)

        logger.info("GRPO trainer initialized with LoRA and Accelerate" if (self.use_lora and self.use_accelerate)
                    else "GRPO trainer initialized with LoRA" if self.use_lora
                    else "GRPO trainer initialized")




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
              is_gspo: bool = False,
              use_Refine_POI_reward: bool = False,
              use_f1_reward: bool = True,
              acc_reward: float = 0.1,
              use_category: bool = True,
              category_reward: float = 0.05,
              ):
        """
        Train the model using GRPO.

        Args:
            text_dataset: Dataset instance for training with textual descriptions.
            output_dir: Output directory for checkpoints.
            run_name: Name of the training run.
            num_train_epochs: Number of training epochs.
            learning_rate: Learning rate.
            per_device_train_batch_size: Batch size per device.
            gradient_accumulation_steps: Number of gradient accumulation steps.
            num_generations: Number of generations per prompt.
            max_prompt_length: Maximum prompt length.
            max_completion_length: Maximum completion length.
            save_steps: Interval for saving checkpoints.
            logging_steps: Interval for logging metrics.
        """
        print(f"\n{'=' * 100}")
        print("Starting GRPO reinforcement learning training")
        print(f"{'=' * 100}")

        # Prepare dataset
        print("Preparing training dataset...")
        dataset = text_dataset

        # Compute total steps
        dataset_size = len(dataset)
        self.total_steps = (dataset_size * num_train_epochs) // (
                    per_device_train_batch_size * gradient_accumulation_steps)

        print(f"Dataset size: {dataset_size}")
        print(f"Epochs: {num_train_epochs}")
        print(f"Batch size: {per_device_train_batch_size}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Estimated total training steps: {self.total_steps}")
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

    # Configure training arguments
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
        # Disable wandb integration
        report_to=None,  # Do not report to any external platform
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
                # Disable wandb integration
                report_to=None,  # Do not report to any external platform
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
        # Define reward functions while capturing reference_responses and progress tracking
        def similarity_reward_func(prompts, completions, reference, **kwargs):
            # Update progress bar
            self.current_step += 1
            self.training_progress.update(1)
            self.training_progress.set_postfix({
                'Step': f"{self.current_step}/{self.total_steps}",
                'Phase': 'Similarity reward computation'
            })

            print(f"\n🔄 Step {self.current_step}/{self.total_steps}: Computing similarity reward")
            similarity_rewards = travel_style_similarity_reward_func(self.similarity_model, prompts, completions,
                                                                     reference, **kwargs)
            # Log to SwanLab
            if SWANLAB_AVAILABLE:
                swanlab.log({
                    "step": self.current_step,
                    "reward/similarity": sum(similarity_rewards) / len(
                        similarity_rewards) if similarity_rewards else 0.0,
                    "progress": (self.current_step / self.total_steps) * 100
                })
            return similarity_rewards

        def length_reward_func(prompts, completions, reference, **kwargs):
            self.training_progress.set_postfix({
                'Step': f"{self.current_step}/{self.total_steps}",
                'Phase': 'Length reward computation'
            })

            print(f"\n📏 Step {self.current_step}/{self.total_steps}: Computing length reward")
            return travel_style_length_reward_func(prompts, completions, reference, **kwargs)

        def quality_reward_func(prompts, completions, reference, **kwargs):
            self.training_progress.set_postfix({
                'Step': f"{self.current_step}/{self.total_steps}",
                'Phase': 'Quality reward computation'
            })

            # Compute and display the total reward
            similarity_rewards = travel_style_similarity_reward_func(self.similarity_model, prompts, completions,
                                                                     reference, **kwargs)
            length_rewards = travel_style_length_reward_func(prompts, completions, reference, **kwargs)

            total_rewards = [s + l for s, l in zip(similarity_rewards, length_rewards)]
            avg_total_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0

            print(f"\n🎯 Step {self.current_step}/{self.total_steps} summary:")
            print(f"   Average total reward: {avg_total_reward:.4f}")
            print(f"   Completion progress: {(self.current_step / self.total_steps) * 100:.1f}%")

            # Log to SwanLab
            if SWANLAB_AVAILABLE:
                swanlab.log({
                    "step": self.current_step,
                    "reward/similarity": sum(similarity_rewards) / len(
                        similarity_rewards) if similarity_rewards else 0.0,
                    "reward/length": sum(length_rewards) / len(length_rewards) if length_rewards else 0.0,
                    "reward/total": avg_total_reward,
                    "progress": (self.current_step / self.total_steps) * 100
                })

            return total_rewards

        def f1_reward_func(prompts, completions, reference, **kwargs):
            self.training_progress.set_postfix({
                'Step': f"{self.current_step}/{self.total_steps}",
                'Phase': 'F1 reward computation'
            })
            if self.f1_evaluator is None:
                logger.error("F1 evaluator unavailable, returning zero rewards")
                return [0.0 for _ in range(len(completions))]
            rewards = travel_style_f1_reward_func(self.f1_evaluator, prompts, completions, reference, **kwargs)
            # Scale the F1 reward by the configured weight
            scaled = [float(r) * float(acc_reward) for r in rewards]
            print(f"Mean raw F1 reward: {sum(rewards)/len(rewards) if rewards else 0.0:.4f} | acc_reward weight={acc_reward} | Mean scaled reward: {sum(scaled)/len(scaled) if scaled else 0.0:.4f}")
            if SWANLAB_AVAILABLE:
                try:
                    swanlab.log({
                        "reward/f1_raw": sum(rewards)/len(rewards) if rewards else 0.0,
                        "reward/f1": sum(scaled)/len(scaled) if scaled else 0.0,
                        "acc_reward": acc_reward,
                        "step": self.current_step
                    })
                except Exception:
                    pass
            return scaled

        def f1_category_reward_func(prompts, completions, reference, **kwargs):
            self.training_progress.set_postfix({
                'Step': f"{self.current_step}/{self.total_steps}",
                'Phase': 'F1 and category reward computation'
            })
            if self.f1_evaluator is None:
                logger.error("F1 evaluator unavailable, returning zero rewards")
                return [0.0 for _ in range(len(completions))]
            f1_rewards, cat_rewards = travel_style_f1_category_reward_func(self.f1_evaluator, prompts, completions, reference, **kwargs)
            # Scale the rewards by their respective weights
            f1_scaled = [float(r) * float(acc_reward) for r in f1_rewards]
            cat_scaled = [float(r) * float(category_reward) for r in cat_rewards]
            scaled = [f + c for f, c in zip(f1_scaled, cat_scaled)]
            print(f"Mean combined reward: {sum(scaled)/len(scaled) if scaled else 0.0:.4f}")
            if SWANLAB_AVAILABLE:
                try:
                    swanlab.log({
                        "reward/f1_raw": sum(f1_rewards)/len(f1_rewards) if f1_rewards else 0.0,
                        "reward/f1": sum(f1_scaled)/len(f1_scaled) if f1_scaled else 0.0,
                        "reward/cat_raw": sum(cat_rewards)/len(cat_rewards) if cat_rewards else 0.0,
                        "reward/cat": sum(cat_scaled)/len(cat_scaled) if cat_scaled else 0.0,
                        "step": self.current_step
                    })
                except Exception:
                    pass
            return scaled
        def Refine_POI_reward_func(prompts, completions, reference, **kwargs):
            self.training_progress.set_postfix({
                'Step': f"{self.current_step}/{self.total_steps}",
                'Phase': 'Refine POI reward computation'
            })
            if self.f1_evaluator is None:
                logger.error("F1 evaluator unavailable, returning zero rewards")
                return [0.0 for _ in range(len(completions))]
            hit_scores, recall_scores, devisity_scores, cat_scores = self.f1_evaluator.compute_Refine_POI_reward(prompts, completions)
            # Scale each reward component by its weight

            hit_scaled = [float(r) * 2.0 for r in hit_scores]
            recall_scaled = [float(r) * 0.5 for r in recall_scores]
            devisity_scaled = [float(r) * 1.0 for r in devisity_scores]
            cat_scaled = [float(r) * 0.5 for r in cat_scores]
            scaled = [h + re + d + c for h, re, d, c in zip(hit_scaled, recall_scaled, devisity_scaled, cat_scaled)]
            print(f"Mean reward after refinement: {sum(scaled)/len(scaled) if scaled else 0.0:.4f}")
            if SWANLAB_AVAILABLE:
                try:
                    swanlab.log({
                        "reward/hit": sum(hit_scaled)/len(hit_scaled) if hit_scaled else 0.0,
                        "reward/recall": sum(recall_scaled)/len(recall_scaled) if recall_scaled else 0.0,
                        "reward/diversity": sum(devisity_scaled)/len(devisity_scaled) if devisity_scaled else 0.0,
                        "reward/cat": sum(cat_scaled)/len(cat_scaled) if cat_scaled else 0.0,
                        "step": self.current_step
                    })
                except Exception:
                    pass
            return scaled


        # Create the GRPO trainer
        if self.use_accelerate:
            # Use accelerate to prepare the model and dataset
            model, dataset = self.accelerator.prepare(self.model, dataset)
            reward_list = [similarity_reward_func]
            if use_f1_reward and use_category:
                reward_list.append(f1_category_reward_func)
            elif use_f1_reward:
                reward_list.append(f1_reward_func)
            trainer = GRPOTrainer(
                model=model,
                processing_class=self.tokenizer,
                reward_funcs=reward_list,
                args=training_args,
                train_dataset=dataset,
            )
        else:
            if self.need_similarity_model:
                reward_list = [similarity_reward_func]
            else:
                reward_list = []
            if use_f1_reward and use_category:
                reward_list.append(f1_category_reward_func)
            elif use_f1_reward:
                reward_list.append(f1_reward_func)
            if use_Refine_POI_reward:
                reward_list = [Refine_POI_reward_func]
            trainer = GRPOTrainer(
                model=self.model,
                processing_class=self.tokenizer,
                reward_funcs=reward_list,
                args=training_args,
                train_dataset=dataset,
            )
            trainer.current_gradient_accumulation_steps = 1  # Keep compatibility with grpo_trainer.py

        logger.info("Starting GRPO training...")
        print("\n🚀 Starting reinforcement learning training...")
        print(f"Model: {self.model_name}")
        print(f"Using LoRA: {'Yes' if self.use_lora else 'No'}")
        if self.use_accelerate:
            print("Using accelerate: Yes")

        # Record training start time
        start_time = time.time()
        if SWANLAB_AVAILABLE:
            swanlab.log({"start_time": start_time})

        try:
            trainer.train()
        except Exception as e:
            print(f"\n❌ Error occurred during training: {e}")
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Ensure the progress bar is closed
            if self.training_progress:
                self.training_progress.close()

        # Compute training duration
        end_time = time.time()
        training_duration = end_time - start_time

        print("\n✅ Training completed!")
        print(f"Training time: {training_duration:.2f} seconds ({training_duration / 60:.1f} minutes)")
        print(f"Total steps: {self.current_step}")

        # Log training completion
        if SWANLAB_AVAILABLE:
            swanlab.log({
                "training_status": "completed",
                "end_time": end_time,
                "training_duration": training_duration,
                "final_step": self.current_step
            })

        # Save the model
        print(f"\n💾 Saving model to: {output_dir}")
        trainer.save_model(output_dir)
        logger.info(f"Model saved to {output_dir}")

        # Log model saving details
        if SWANLAB_AVAILABLE:
            swanlab.log({
                "model_saved": True,
                "model_path": output_dir,
                "final_epoch": num_train_epochs
            })

        print(f"\n{'=' * 100}")
        print("GRPO reinforcement learning training finished!")
        print(f"{'=' * 100}\n")

    # ================= Supervised Fine-Tuning (SFT) ================= #
    def sft(self,
            text_dataset=None,
            output_dir: str = "./sft_travel_style_lora",
            run_name: str = "travel_style_sft",
            num_train_epochs: int = 2,
            learning_rate: float = 2e-5,
            per_device_train_batch_size: int = 1,
            gradient_accumulation_steps: int = 8,
            max_seq_length: int = 2048,
            logging_steps: int = 10,
            save_steps: int = 500,
            packing: bool = False,
            swanlab_project: str = "travel-style-sft",
            enable_swanlab: bool = True,
            use_accelerate: bool = False
            ):
        """Perform supervised fine-tuning (SFT) on the LLM.

        The dataset must contain the following fields:
            - prompt: Input prompt (str)
            - reference: Target output (str)

        After training, only the LoRA adapter parameters are saved (if LoRA is enabled), keeping the
        base model path unchanged.

        Args:
            text_dataset: HuggingFace dataset or List[Dict] containing prompt/reference pairs.
            output_dir: Directory to save the LoRA adapter.
            run_name: Name of the run.
            num_train_epochs: Number of training epochs.
            learning_rate: Learning rate.
            per_device_train_batch_size: Batch size per device.
            gradient_accumulation_steps: Number of gradient accumulation steps.
            max_seq_length: Maximum sequence length (after truncation).
            logging_steps: Logging interval during training.
            save_steps: Checkpoint saving interval.
            packing: Whether to enable multi-sample packing (SFTTrainer parameter).
        """

        logger.info("Starting SFT fine-tuning ...")
        print(f"\n{'=' * 90}\nStarting supervised fine-tuning (SFT)\n{'=' * 90}")

        from datasets import Dataset
        def prepare_dataset(ds):

            """
            ds: A HuggingFace Dataset object containing "prompt" and "reference" columns.

            Returns: A transformed dataset where each record contains "prompt" and "completion",
            or pre-concatenated text that matches the format expected by SFTTrainer.
            """
            def convert_example(ex):
                # Convert prompt + reference into the format accepted by SFTTrainer
                # TRL supports the {"prompt": ..., "completion": ...} format :contentReference[oaicite:1]{index=1}
                return {
                    "prompt": ex["prompt"],
                    "completion": ex["reference"],
                }

            new_ds = ds.map(convert_example, remove_columns=[col for col in ds.column_names if col not in ("prompt", "reference")])
            return new_ds

        formatted_dataset = prepare_dataset(text_dataset)

    # Build TrainingArguments (separate from GRPO to avoid conflicts)
        from transformers import TrainingArguments
        sft_args = TrainingArguments(
            output_dir=output_dir,
            run_name=run_name,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_steps=save_steps,
            bf16=True,
            report_to=[]  # Disable wandb and other reporters
        )

        # Create the SFTTrainer
        # SwanLab callback (only when available and enabled)
        swanlab_callback = None
        if enable_swanlab and SWANLAB_AVAILABLE:
            try:
                class SwanLabSFTCallback(TrainerCallback):
                    def __init__(self):
                        self.started = False
                    def on_train_begin(self, args, state, control, **kwargs):
                        if not self.started:
                            swanlab.log({"event": "sft_train_begin", "total_steps": state.max_steps})
                            self.started = True
                    def on_log(self, args, state, control, logs=None, **kwargs):
                        if logs:
                            # Filter out unsupported types
                            safe_logs = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}
                            if safe_logs:
                                swanlab.log({f"sft/{k}": v for k, v in safe_logs.items()})
                    def on_step_end(self, args, state, control, **kwargs):
                        if state.global_step % max(1, logging_steps) == 0:
                            swanlab.log({"sft/step": state.global_step, "sft/epoch_progress": state.epoch or 0})
                    def on_train_end(self, args, state, control, **kwargs):
                        swanlab.log({"event": "sft_train_end", "final_step": state.global_step, "final_epoch": state.epoch or 0})
                swanlab.init(project=swanlab_project, experiment_name=run_name, config={
                    "mode": "sft",
                    "model_name": self.model_name,
                    "use_lora": self.use_lora,
                    "epochs": num_train_epochs,
                    "lr": learning_rate,
                    "batch_size": per_device_train_batch_size,
                    "grad_accum": gradient_accumulation_steps,
                    "max_seq_length": max_seq_length,
                    "samples": len(text_dataset)
                })
                swanlab_callback = SwanLabSFTCallback()
            except Exception as e:
                logger.warning(f"Failed to initialize SwanLab, continuing without logging: {e}")
                swanlab_callback = None

        sft_trainer = SFTTrainer(
            model=self.model,
            train_dataset=formatted_dataset,
            args=sft_args,
            callbacks=[swanlab_callback] if swanlab_callback else None,
        )
        # SFTTrainer internally handles accelerator usage
        # sft_trainer, formatted_dataset = self.accelerator.prepare(sft_trainer, formatted_dataset)

        logger.info("SFT Trainer initialized, starting training ...")
        print(f"Samples: {len(formatted_dataset)} | Epochs: {num_train_epochs} | batch: {per_device_train_batch_size} | accumulation: {gradient_accumulation_steps}")
        try:
            sft_trainer.train()
        except Exception as e:
            logger.error(f"SFT training failed: {e}")
            raise

        # Save only the LoRA adapter
        os.makedirs(output_dir, exist_ok=True)
        if self.use_lora:
            try:
                self.model.save_pretrained(output_dir)
                logger.info(f"LoRA adapter saved to {output_dir}")
            except Exception as e:
                logger.error(f"Failed to save LoRA adapter: {e}")
                raise
        else:
            # If LoRA is not used, save the entire model (honoring the user's path requirement)
            self.model.save_pretrained(output_dir)
            logger.info(f"LoRA disabled, full model saved to {output_dir}")

        # Optionally save the tokenizer
        try:
            self.tokenizer.save_pretrained(output_dir)
        except Exception:
            pass

        print(f"\n✅ SFT completed, adapter stored at: {output_dir}\n")
        if enable_swanlab and SWANLAB_AVAILABLE:
            try:
                swanlab.log({"sft/adapter_saved": True, "sft/adapter_path": output_dir})
                print("SwanLab logs updated. Example command for local UI:\n  swanlab board\nOr filter by experiment_name=", run_name)
            except Exception:
                pass
        return output_dir


def main():
    import wandb
    import argparse
    wandb.init(mode="disabled")  # Explicitly disable wandb
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_accelerate", action="store_true", help="Enable accelerate for training")
    # SFT-specific arguments
    parser.add_argument("--sft", action="store_true", help="Run supervised fine-tuning (SFT)")
    parser.add_argument("--sft_epochs", type=int, default=1, help="Number of epochs for SFT")
    parser.add_argument("--sft_lr", type=float, default=2e-5, help="Learning rate for SFT")
    parser.add_argument("--sft_batch", type=int, default=2, help="Per-device batch size for SFT")
    parser.add_argument("--sft_grad_accum", type=int, default=4, help="Gradient accumulation steps for SFT")
    parser.add_argument("--sft_max_len", type=int, default=2048, help="Maximum sequence length for SFT")
    parser.add_argument("--sft_output", type=str, default="./sft_travel_style_lora", help="Output directory for the SFT LoRA adapter")
    parser.add_argument("--sft_run_name", type=str, default="travel_style_sft", help="Name of the SFT run")
    parser.add_argument("--sft_project", type=str, default="travel-style-sft", help="SwanLab project name for SFT")
    parser.add_argument("--no_swanlab", action="store_true", help="Disable SwanLab logging during SFT")
    parser.add_argument("--dataset_name", type=str, default="Foursquare", help="Dataset name (Foursquare/Yelp)")
    args = parser.parse_args()
    dataset_name = args.dataset_name
    args.sft_output = args.sft_output + f"_{dataset_name}_sftepoch1"
    args.sft_project = args.sft_project + f"_{dataset_name}_sftepoch1"
    args.sft_run_name = args.sft_run_name + f"_{dataset_name}_sftepoch1"

    from datasets import load_from_disk
    if dataset_name == "Foursquare":
        text_dataset = load_from_disk("../dataset/travel_dataset_20250712_201017")
        output_dir = "./sft_grpo_Foursquare_f1_RefinePOI_newlora"
        run_name = "Foursquare_sft_grpo_RefinePOI_newlora"
        lora_config = "./sft_travel_style_lora_Foursquare_sftepoch1/checkpoint-376"
    else:
        text_dataset = load_from_disk("../dataset/Yelp_20250714_192438")
        output_dir = "./sft_grpo_Yelp_f1_epoch1_RefinePOI_newlora"
        run_name = "Yelp_sft_grpo_epoch1_RefinePOI_newlora"
        lora_config = "./sft_travel_style_lora_Yelp_sftepoch1/checkpoint-553"
    trainer = TravelStyleGRPOTrainer(
        model_name="../LLMs/Qwen3-8B",
        model_run_name=dataset_name + "_semantic8_diversity0.1_attnreg0.1_mask0.75",
        dataset_name=dataset_name,
        use_lora=True,
        lora_config=lora_config if not args.sft else None,
        lora_r=16,
        lora_alpha=32,
        is_sft=args.sft,
        use_accelerate=args.use_accelerate,
        need_similarity_model=False
    )

    if args.sft:
        adapter_dir = trainer.sft(
            text_dataset=text_dataset,
            output_dir=args.sft_output,
            run_name=args.sft_run_name,
            num_train_epochs=args.sft_epochs,
            learning_rate=args.sft_lr,
            per_device_train_batch_size=args.sft_batch,
            gradient_accumulation_steps=args.sft_grad_accum,
            max_seq_length=args.sft_max_len,
            enable_swanlab=not args.no_swanlab,
            swanlab_project=args.sft_project,
        )

    else:
        # RL for training
        trainer.train(
            text_dataset=text_dataset,
            output_dir=output_dir,
            run_name=run_name,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            use_Refine_POI_reward=True,
            use_category=True,
            use_f1_reward=True,
            is_gspo=False
        )

    if SWANLAB_AVAILABLE:
        swanlab.finish()
        print("SwanLab experiment finished")


if __name__ == "__main__":
    main()
