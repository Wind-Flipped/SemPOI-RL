# -*- coding: UTF-8 -*-
"""
LLMs.py - 大语言模型调用接口和强化学习训练模块

该模块包含：
1. 调用大语言模型生成旅游风格的接口
2. 使用GRPO方法对LLM进行强化学习的训练接口
3. 文本相似度计算和准确率计算的奖励函数
4. F1-Score奖励：将RL生成的文本作为目标LLM输入到保存的SPOTModel中，
    对应样本运行推理得到预测轨迹，与真实轨迹计算 sample_f1 作为奖励
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'  # 设置可见GPU设备
import sys

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments
)
from transformers.trainer_callback import TrainerCallback
from trl import GRPOConfig, GRPOTrainer, SFTTrainer
# from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from data import TravelDataset
from tqdm import tqdm
import time
# from vllm import LLM, SamplingParams
import logging
import pickle
from metrics import category_consistency_rate, hit_rate, recall_rate, diversity_rate
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

# ===== 为F1奖励加载SPOT-Trip模型与数据所需依赖 =====
from data import TravelDataset, random_split  # 数据集
from utils import collate_fn  # DataLoader的聚合函数
from model import SPOTModel  # 主模型

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 独立的奖励函数（参照test.py的格式）
def travel_style_similarity_reward_func(similarity_model, prompts, completions, reference_responses, **kwargs) -> list[
    float]:
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

    print(f"\n{'=' * 80}")
    print(f"相似度奖励函数计算 - 处理 {len(responses)} 个生成结果")
    print(f"{'=' * 80}")

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
                print(f"\n样本 {i + 1}:")
                print(f"Prompt: {prompts[i][:100]}..." if len(prompts[i]) > 100 else f"Prompt: {prompts[i]}")
                print(f"生成回复: {response[:150]}..." if len(response) > 150 else f"生成回复: {response}")
                print(f"参考答案: {reference[:150]}..." if len(reference) > 150 else f"参考答案: {reference}")
                print(f"相似度: {similarity:.4f}")
                print(f"相似度奖励: {reward:.4f}")
                print(f"{'-' * 60}")

            except Exception as e:
                logger.error(f"Error calculating similarity reward: {e}")
                rewards.append(0.0)
                print(f"样本 {i + 1}: 相似度计算错误，奖励设为 0.0")
        else:
            rewards.append(0.0)
            print(f"样本 {i + 1}: 缺少参考答案，奖励设为 0.0")

    avg_similarity_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"\n平均相似度奖励: {avg_similarity_reward:.4f}")
    print(f"{'=' * 80}\n")

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

    print(f"\n{'=' * 80}")
    print(f"长度奖励函数计算 - 处理 {len(responses)} 个生成结果")
    print(f"{'=' * 80}")

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
                print(f"样本 {i + 1}:")
                print(f"生成文本长度: {predicted_length} 词")
                print(f"参考文本长度: {reference_length} 词")
                print(f"长度差异比率: {length_diff:.4f}")
                print(f"长度奖励: {length_reward:.4f}")
                print(f"{'-' * 60}")
            else:
                rewards.append(0.0)
                print(f"样本 {i + 1}: 参考答案长度为0，奖励设为 0.0")
        else:
            rewards.append(0.0)
            print(f"样本 {i + 1}: 缺少参考答案，奖励设为 0.0")

    avg_length_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"\n平均长度奖励: {avg_length_reward:.4f}")
    print(f"{'=' * 80}\n")

    return rewards


# ===================== 基于SPOTModel的F1奖励 ===================== #
class _RLArgsStub:
    """最小化参数对象，满足SPOTModel推理所需字段。"""
    # 与main.py中的默认值保持一致，必要时可在TravelStyleGRPOTrainer初始化时覆盖
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
    延迟加载SPOTModel检查点与数据集，并将RL生成文本作为messages输入模型，
    计算与真实目的地轨迹的样本级F1（sample_f1），作为奖励返回。
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
            logger.log(f"[warn] 无法加载 poi_meta.pkl: {_e}")

        self._data = None
        self._model = None
        self._offset = 0  # 与RL批次对齐的指针
        self._dl_cache = {}
        # 直接加载（不延迟）
        self._lazy_load_data_and_model()

    # ---------- 内部工具 ---------- #
    def _find_checkpoint(self) -> str:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../{self.dataset_name}/model_save/{self.run_name}"))
        if not os.path.isdir(base):
            raise FileNotFoundError(f"未找到模型目录: {base}")
        if self.dataset_name == "Foursquare":
            best = os.path.join(base, "model_0.xhr")
        else:
            best = os.path.join(base, "model_5.xhr")
        if os.path.exists(best):
            return best
        else:
            raise FileNotFoundError(f"目录中未找到检查点: {base}")


    def _lazy_load_data_and_model(self):
        # 已由构造函数主动调用，不再延迟；若已加载则直接返回
        if self._data is not None and self._model is not None:
            return

        # 加载原始轨迹数据（与main.py保持一致路径约定）
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

        # 构建模型（和main.py一致）
        max_d_length = max(len(seq) for seq in self._data.dsts)
        max_o_length = max(len(seq) for seq in self._data.oris)

        model = SPOTModel(
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

        # 加载权重
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
                # 兼容OpenAI/TRL格式: [{"content": "..."}, ...]
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
        # 移除padding 0
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
        # 移除padding 0
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
        基于给定的样本下标列表，返回与原 loader 完全一致顺序与批次边界的 DataLoader：
        - 使用 Subset 保留给定 indices 的相对顺序（无需再 shuffle）
        - 使用相同的 collate_fn
        - batch_size 由调用方提供，从而复现与原批次相同的切分

        注意：PyTorch 的 Subset 会按照传入 indices 的顺序进行索引；
        DataLoader 在 shuffle=False 时会按顺序遍历 Subset，因而能保证批次一致。
        """
        from torch.utils.data import Subset, DataLoader

        if not indices:
            # 空索引，直接返回一个空的 DataLoader（不会产生任何 batch）
            return DataLoader([], batch_size=1)

        subset = Subset(self._data, indices)
        # 关闭 shuffle，严格按 indices 的顺序迭代；collate_fn 与原训练保持一致
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        return loader

    def compute_batch_f1(self, prompts, completions) -> list[float]:
        """
        给定一批次RL生成文本，取数据集中对应顺序的样本，
        将文本作为messages输入模型进行推理，返回每个样本的sample_f1。
        """
        self._lazy_load_data_and_model()
        msgs = self._tensorize_messages(completions)

        bsz = len(msgs)
        N = len(self._data)
        # 取连续下标，必要时取模
        idxs = [ (self._offset + i) % N for i in range(bsz) ]
        self._offset = (self._offset + bsz) % N

        loader = self._get_subset_loader(idxs, batch_size=bsz)

        f1_scores: list[float] = []
        # 遍历一个batch（期望只有1个，因为batch_size=bsz）
        with torch.no_grad():
            for (uid, o_ck, d_ck, masked_d_ck, o_h, d_h, masked_d_h, o_t, d_t, o_l, d_l, o_pad, d_pad, o_rg, d_rg) in loader:
                # 将messages列表传入forward
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

                # 预测
                predicted_ids = self._model(uid, msgs, o_ck, masked_d_ck, o_t, d_t, o_l, d_l, o_pad, d_pad, d_ck, o_rg,
                                      d_rg, target_seq=None)


                # 分样本计算F1
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

        # 与输入数量对齐（理论上相等）
        if len(f1_scores) < bsz:
            f1_scores += [0.0] * (bsz - len(f1_scores))
        elif len(f1_scores) > bsz:
            f1_scores = f1_scores[:bsz]
        return f1_scores

    def compute_batch_f1_category(self, prompts, completions) -> list[float]:
        """
        给定一批次RL生成文本，取数据集中对应顺序的样本，
        将文本作为messages输入模型进行推理，返回每个样本的sample_f1和类别一致率。
        """
        self._lazy_load_data_and_model()
        msgs = self._tensorize_messages(completions)

        bsz = len(msgs)
        N = len(self._data)
        # 取连续下标，必要时取模
        idxs = [ (self._offset + i) % N for i in range(bsz) ]
        self._offset = (self._offset + bsz) % N

        loader = self._get_subset_loader(idxs, batch_size=bsz)

        f1_scores: list[float] = []
        cat_scores: list[float] = []
        # 遍历一个batch（期望只有1个，因为batch_size=bsz）
        with torch.no_grad():
            for (uid, o_ck, d_ck, masked_d_ck, o_h, d_h, masked_d_h, o_t, d_t, o_l, d_l, o_pad, d_pad, o_rg, d_rg) in loader:
                # 将messages列表传入forward
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

                # 预测
                predicted_ids = self._model(uid, msgs, o_ck, masked_d_ck, o_t, d_t, o_l, d_l, o_pad, d_pad, d_ck, o_rg,
                                      d_rg, target_seq=None)


                # 分样本计算F1
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
                    # 计算类别一致率
                    cat_rate = category_consistency_rate(sample_pred, sample_target, self.poi_meta)
                    cat_scores.append(cat_rate)

        # 与输入数量对齐（理论上相等）
        if len(f1_scores) < bsz:
            f1_scores += [0.0] * (bsz - len(f1_scores))
        elif len(f1_scores) > bsz:
            f1_scores = f1_scores[:bsz]
        return f1_scores, cat_scores

    def compute_Refine_POI_reward(self, prompts, completions) -> list[float]:
        """
        给定一批次RL生成文本，取数据集中对应顺序的样本，
        将文本作为messages输入模型进行推理，返回每个样本的sample_f1和类别一致率。
        """
        self._lazy_load_data_and_model()
        msgs = self._tensorize_messages(completions)

        bsz = len(msgs)
        N = len(self._data)
        # 取连续下标，必要时取模
        idxs = [ (self._offset + i) % N for i in range(bsz) ]
        self._offset = (self._offset + bsz) % N

        loader = self._get_subset_loader(idxs, batch_size=bsz)

        hit_scores: list[float] = []
        recall_scores: list[float] = []
        devisity_scores: list[float] = []
        cat_scores: list[float] = []
        # 遍历一个batch（期望只有1个，因为batch_size=bsz）
        with torch.no_grad():
            for (uid, o_ck, d_ck, masked_d_ck, o_h, d_h, masked_d_h, o_t, d_t, o_l, d_l, o_pad, d_pad, o_rg, d_rg) in loader:
                # 将messages列表传入forward
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

                # 预测
                predicted_ids = self._model(uid, msgs, o_ck, masked_d_ck, o_t, d_t, o_l, d_l, o_pad, d_pad, d_ck, o_rg,
                                      d_rg, target_seq=None)


                # 分样本计算F1
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
                    # 计算类别一致率
                    cat_rate = category_consistency_rate(sample_pred, sample_target, self.poi_meta)
                    cat_scores.append(cat_rate)
                    recall = recall_rate(sample_pred, sample_target)
                    recall_scores.append(recall)
                    diversity = diversity_rate(sample_pred)
                    devisity_scores.append(diversity)



        return hit_scores, recall_scores, devisity_scores, cat_scores

def travel_style_f1_reward_func(evaluator: F1RewardEvaluator, prompts, completions, reference_responses, **kwargs) -> list[float]:
    """
    基于保存的SPOTModel计算样本级F1作为奖励。
    返回值范围[0,1]，与trainer.py中的sample_f1一致语义。
    """
    # 直接调用评估器
    # try:
    #     rewards = evaluator.compute_batch_f1(prompts, completions)
    # except Exception as e:
    #     print(f"[F1-Reward] 计算失败: {e}")
    #     rewards = [0.0 for _ in range(len(completions))]
    rewards = evaluator.compute_batch_f1(prompts, completions)
    # 打印摘要
    if rewards:
        print(f"F1奖励(均值): {sum(rewards)/len(rewards):.4f} | 样本数: {len(rewards)}")
    return rewards

def travel_style_f1_category_reward_func(evaluator: F1RewardEvaluator, prompts, completions, reference_responses, **kwargs) -> list[float]:
    """
    基于保存的SPOTModel计算样本级F1作为奖励。
    返回值范围[0,1]，与trainer.py中的sample_f1一致语义。
    """
    # 直接调用评估器
    # try:
    #     rewards = evaluator.compute_batch_f1(prompts, completions)
    # except Exception as e:
    #     print(f"[F1-Reward] 计算失败: {e}")
    #     rewards = [0.0 for _ in range(len(completions))]
    f1_rewards, cat_rewards = evaluator.compute_batch_f1_category(prompts, completions)
    # 打印摘要
    if f1_rewards:
        print(f"f1奖励(均值): {sum(f1_rewards)/len(f1_rewards):.4f} | 样本数: {len(f1_rewards)}")
    if cat_rewards:
        print(f"类别一致率奖励(均值): {sum(cat_rewards)/len(cat_rewards):.4f} | 样本数: {len(cat_rewards)}")
    return f1_rewards, cat_rewards

class TravelStyleGRPOTrainer:
    """使用GRPO方法训练旅游风格生成模型"""

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
        self.lora_config = lora_config
        # 用于F1奖励的SPOT-Trip配置
        self.dataset_name = dataset_name
        self.model_run_name = model_run_name
        self.need_similarity_model = need_similarity_model

        # 训练进度跟踪
        self.training_progress = None
        self.current_step = 0
        self.total_steps = 0

        # F1奖励评估器：在cuda:1上提前初始化
        if not is_sft:
            try:
                self.f1_evaluator = F1RewardEvaluator(
                    dataset_name=self.dataset_name,
                    run_name=self.model_run_name,
                    device="cuda:1",
                )
            except Exception as _e:
                logger.error(f"预加载F1RewardEvaluator失败: {_e}")
                self.f1_evaluator = None

        if is_train:
            # 初始化相似度模型
            if not is_sft and need_similarity_model:
                from sentence_transformers import SentenceTransformer
                self.similarity_model = SentenceTransformer(similarity_model_name, device="cuda:3")

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
                    # max_memory={0: "20GiB", 1: "20GiB"},
                    torch_dtype=torch.bfloat16
                )
            for param in self.model.parameters():
                param.requires_grad = False  # freeze the model - train adapters later

            # 配置LoRA
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
                    # Add a new lora adapter
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
        print(f"\n{'=' * 100}")
        print(f"开始GRPO强化学习训练")
        print(f"{'=' * 100}")

        # 准备数据集
        print("准备训练数据集...")
        dataset = text_dataset

        # 计算总步数
        dataset_size = len(dataset)
        self.total_steps = (dataset_size * num_train_epochs) // (
                    per_device_train_batch_size * gradient_accumulation_steps)

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

        # 配置训练参数
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
            similarity_rewards = travel_style_similarity_reward_func(self.similarity_model, prompts, completions,
                                                                     reference, **kwargs)
            # 记录到SwanLab
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
            similarity_rewards = travel_style_similarity_reward_func(self.similarity_model, prompts, completions,
                                                                     reference, **kwargs)
            length_rewards = travel_style_length_reward_func(prompts, completions, reference, **kwargs)

            total_rewards = [s + l for s, l in zip(similarity_rewards, length_rewards)]
            avg_total_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0

            print(f"\n🎯 步骤 {self.current_step}/{self.total_steps} 总结:")
            print(f"   平均总奖励: {avg_total_reward:.4f}")
            print(f"   完成进度: {(self.current_step / self.total_steps) * 100:.1f}%")

            # 记录到SwanLab
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
                'Phase': 'F1奖励计算'
            })
            if self.f1_evaluator is None:
                logger.error("F1评估器不可用，返回0奖励")
                return [0.0 for _ in range(len(completions))]
            rewards = travel_style_f1_reward_func(self.f1_evaluator, prompts, completions, reference, **kwargs)
            # 按权重缩放F1奖励
            scaled = [float(r) * float(acc_reward) for r in rewards]
            print(f"F1原始奖励均值: {sum(rewards)/len(rewards) if rewards else 0.0:.4f} | 权重acc_reward={acc_reward} | 缩放后均值: {sum(scaled)/len(scaled) if scaled else 0.0:.4f}")
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
                'Phase': 'F1与类别奖励计算'
            })
            if self.f1_evaluator is None:
                logger.error("F1评估器不可用，返回0奖励")
                return [0.0 for _ in range(len(completions))]
            f1_rewards, cat_rewards = travel_style_f1_category_reward_func(self.f1_evaluator, prompts, completions, reference, **kwargs)
            # 按权重缩放F1奖励
            f1_scaled = [float(r) * float(acc_reward) for r in f1_rewards]
            cat_scaled = [float(r) * float(category_reward) for r in cat_rewards]
            scaled = [f + c for f, c in zip(f1_scaled, cat_scaled)]
            print(f"预测结果奖励均值: {sum(scaled)/len(scaled) if scaled else 0.0:.4f}")
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
                'Phase': 'F1与类别奖励计算'
            })
            if self.f1_evaluator is None:
                logger.error("F1评估器不可用，返回0奖励")
                return [0.0 for _ in range(len(completions))]
            hit_scores, recall_scores, devisity_scores, cat_scores = self.f1_evaluator.compute_Refine_POI_reward(prompts, completions)
            # 按权重缩放F1奖励

            hit_scaled = [float(r) * 2.0 for r in hit_scores]
            recall_scaled = [float(r) * 0.5 for r in recall_scores]
            devisity_scaled = [float(r) * 1.0 for r in devisity_scores]
            cat_scaled = [float(r) * 0.5 for r in cat_scores]
            scaled = [h + re + d + c for h, re, d, c in zip(hit_scaled, recall_scaled, devisity_scaled, cat_scaled)]
            print(f"预测结果奖励均值: {sum(scaled)/len(scaled) if scaled else 0.0:.4f}")
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


        # 创建GRPO训练器
        if self.use_accelerate:
            # 使用accelerate准备模型和数据集
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
            trainer.current_gradient_accumulation_steps = 1  # 与grpo_trainer.py兼容

        logger.info("Starting GRPO training...")
        print(f"\n🚀 开始强化学习训练...")
        print(f"模型: {self.model_name}")
        print(f"使用LoRA: {'是' if self.use_lora else '否'}")
        if self.use_accelerate:
            print(f"使用 accelerate: 是")

        # 记录训练开始时间
        start_time = time.time()
        if SWANLAB_AVAILABLE:
            swanlab.log({"start_time": start_time})

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
        print(f"训练用时: {training_duration:.2f} 秒 ({training_duration / 60:.1f} 分钟)")
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

        print(f"\n{'=' * 100}")
        print(f"GRPO强化学习训练完成!")
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
        """对LLM进行有监督微调 (Supervised Fine-Tuning)。

        要求数据集中含有以下字段：
            - prompt: 输入提示 (str)
            - reference: 目标输出 (str)

        训练完成后仅保存 LoRA 适配器参数（如果开启 LoRA），保持基座模型路径不变。

        Args:
            text_dataset: HuggingFace datasets 或 List[Dict]，包含 prompt/reference
            output_dir: LoRA 适配器保存目录
            run_name: 运行名称
            num_train_epochs: 训练轮数
            learning_rate: 学习率
            per_device_train_batch_size: 每设备 batch size
            gradient_accumulation_steps: 梯度累积
            max_seq_length: 最大序列长度（截断）
            logging_steps: 日志步频
            save_steps: 保存步频
            packing: 是否启用多样本打包（SFTTrainer参数）
        """

        logger.info("开始 SFT 微调 ...")
        print(f"\n{'=' * 90}\n开始监督微调 (SFT)\n{'=' * 90}")

        from datasets import Dataset
        def prepare_dataset(ds):

            """
            ds: 一个 HuggingFace Dataset 对象，包含 "prompt" 和 "reference" 两列

            返回：一个 transform 后的 dataset，其每条 record 包含 "prompt" 和 "completion" 或者直接是拼接好的文本，满足 SFTTrainer 期待的格式
            """
            def convert_example(ex):
                # 这里把 prompt + reference 转成 SFTTrainer 可接受的格式
                # TRL 支持 “{"prompt": ..., "completion": ...}” 格式 :contentReference[oaicite:1]{index=1}
                return {
                    "prompt": ex["prompt"],
                    "completion": ex["reference"],
                }

            new_ds = ds.map(convert_example, remove_columns=[col for col in ds.column_names if col not in ("prompt", "reference")])
            return new_ds

        formatted_dataset = prepare_dataset(text_dataset)

        # 构造 TrainingArguments (与 GRPO 分离，避免冲突)
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
            report_to=[]  # 禁用wandb等
        )

        # 创建 SFTTrainer
        # SwanLab 回调（仅在可用且启用时）
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
                            # 过滤掉不适合的对象类型
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
                logger.warning(f"SwanLab 初始化失败，继续训练: {e}")
                swanlab_callback = None

        sft_trainer = SFTTrainer(
            model=self.model,
            train_dataset=formatted_dataset,
            args=sft_args,
            callbacks=[swanlab_callback] if swanlab_callback else None,
        )
        # SFTTrainer 内部会处理 accelerator
        # sft_trainer, formatted_dataset = self.accelerator.prepare(sft_trainer, formatted_dataset)

        logger.info("SFT Trainer 初始化完成，开始训练 ...")
        print(f"样本数: {len(formatted_dataset)} | 轮数: {num_train_epochs} | batch: {per_device_train_batch_size} | 累积: {gradient_accumulation_steps}")
        try:
            sft_trainer.train()
        except Exception as e:
            logger.error(f"SFT 训练失败: {e}")
            raise

        # 仅保存 LoRA 适配器
        os.makedirs(output_dir, exist_ok=True)
        if self.use_lora:
            try:
                self.model.save_pretrained(output_dir)
                logger.info(f"LoRA 适配器已保存到 {output_dir}")
            except Exception as e:
                logger.error(f"保存 LoRA 适配器失败: {e}")
                raise
        else:
            # 如果未使用LoRA，也保存整个模型（用户要求保持路径不变，此处给出提示）
            self.model.save_pretrained(output_dir)
            logger.info(f"未使用LoRA，已保存全量模型到 {output_dir}")

        # 保存 tokenizer（可选）
        try:
            self.tokenizer.save_pretrained(output_dir)
        except Exception:
            pass

        print(f"\n✅ SFT 完成，适配器保存在: {output_dir}\n")
        if enable_swanlab and SWANLAB_AVAILABLE:
            try:
                swanlab.log({"sft/adapter_saved": True, "sft/adapter_path": output_dir})
                print("SwanLab 日志已更新。查看命令示例 (本地 UI)：\n  swanlab board\n或在项目面板中筛选 experiment_name=", run_name)
            except Exception:
                pass
        return output_dir


def main():
    """主函数 - 演示使用方法"""
    import wandb
    import argparse
    wandb.init(mode="disabled")  # 强制禁用 wandb
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_accelerate", action="store_true", help="是否使用accelerate加速")
    # SFT 相关
    parser.add_argument("--sft", action="store_true", help="是否执行监督微调 (SFT)")
    parser.add_argument("--sft_epochs", type=int, default=1, help="SFT 训练轮数")
    parser.add_argument("--sft_lr", type=float, default=2e-5, help="SFT 学习率")
    parser.add_argument("--sft_batch", type=int, default=2, help="SFT per-device batch size")
    parser.add_argument("--sft_grad_accum", type=int, default=4, help="SFT 梯度累积步数")
    parser.add_argument("--sft_max_len", type=int, default=2048, help="SFT 最大序列长度")
    parser.add_argument("--sft_output", type=str, default="./sft_travel_style_lora", help="SFT LoRA保存目录")
    parser.add_argument("--sft_run_name", type=str, default="travel_style_sft", help="SFT 运行名称")
    parser.add_argument("--sft_project", type=str, default="travel-style-sft", help="SwanLab 项目名称")
    parser.add_argument("--no_swanlab", action="store_true", help="禁用 SFT 中的 SwanLab 日志")
    parser.add_argument("--dataset_name", type=str, default="Foursquare", help="数据集名称(Foursquare/Yelp)")
    args = parser.parse_args()
    dataset_name = args.dataset_name
    args.sft_output = args.sft_output + f"_{dataset_name}_sftepoch1"
    args.sft_project = args.sft_project + f"_{dataset_name}_sftepoch1"
    args.sft_run_name = args.sft_run_name + f"_{dataset_name}_sftepoch1"
    # 初始化训练器（使用LoRA和accelerate）

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
        use_accelerate=args.use_accelerate,  # 启用accelerate支持
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
        # 打印 SwanLab 查看命令
        if not args.no_swanlab and SWANLAB_AVAILABLE:
            print("\n=== SwanLab 查看方式 ===")
            print("1) 启动本地面板: swanlab board")
            print(f"2) 进入项目: {args.sft_project}")
            print(f"3) 过滤 experiment_name == {args.sft_run_name}")
            print("4) 关键指标前缀: sft/* \n")
            print(f"LoRA 适配器路径: {adapter_dir}")
        else:
            print("SwanLab 未启用或不可用，跳过日志查看说明。")
    else:
        # 默认 GRPO 训练流程

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

    # 评估模型
    # results = trainer.evaluate(test_trajectories)
    # print(f"Evaluation results: {results}")

    # 结束SwanLab实验记录
    if SWANLAB_AVAILABLE:
        swanlab.finish()
        print("SwanLab experiment finished")


if __name__ == "__main__":
    main()
