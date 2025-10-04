# -*- coding: utf-8 -*-
from ast import parse
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"  # Set the visible GPU device
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'  # 设置可见GPU设备
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

import argparse
from collections import namedtuple, defaultdict
import numpy as np
import os
import sys
from copy import copy
import warnings

warnings.filterwarnings('ignore')
try:
    import ipdb
except:
    pass

from utils import *
from data import TravelDataset, random_split, KGDataset
from ARmodel import ARModel
from model import SPOTModel
import metrics
from trainer import *

import pickle


def main():
    parser = argparse.ArgumentParser()
    # Dataset arguments
    dataset_name = 'Yelp'  # Default dataset name, can be changed to 'Yelp'
    parser.add_argument('--dataset_name', type=str, default=dataset_name, choices=['Foursquare', 'Yelp'])
    parser.add_argument('--ori_data', type=str, default=f'../{dataset_name}/home.txt')
    parser.add_argument('--dst_data', type=str, default=f'../{dataset_name}/oot.txt')
    parser.add_argument('--trans_data', type=str, default=f'../{dataset_name}/travel.txt')
    parser.add_argument('--save_path', type=str, default=f'../{dataset_name}/model_save')
    parser.add_argument("--best_save", action="store_true")
    parser.add_argument("--kg_path", type=str, default=f'../{dataset_name}/kg.txt')
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--data_split_path', type=str, default=f'../{dataset_name}/data_split.pkl')

    # Training Configurations
    parser.add_argument('--model', type=str, default='SPOT-Trip')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--train_batch', type=int, default=4)
    parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument('--test_batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument("--projection_dim", type=int, default=64)

    parser.add_argument('--margin', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--lr_dc', type=float, default=0.2)
    parser.add_argument('--lr_dc_step', type=int, default=4)
    parser.add_argument('--l2', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=2050)
    parser.add_argument('--log_path', type=str, default='../')
    parser.add_argument('--log', action="store_true")
    parser.add_argument('--name', type=str, default="default")
    # parser.add_argument('--model', type=str, default="base")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument("--stop_epoch", type=int, default=2)  # early stopping
    parser.add_argument("--fine_stop", type=int, default=12)

    # Knowledge Graph (KG) Arguments
    parser.add_argument("--segments", type=int, default=16)
    parser.add_argument("--kg", action="store_true")
    parser.add_argument("--entity_num_per_poi", type=int, default=2)  # Note: F 2 For Yelp, use 10
    parser.add_argument("--train_trans", action="store_true")
    parser.add_argument('--trans', type=str, default="transe")
    parser.add_argument("--contrast", action="store_true")
    parser.add_argument("--kgcn", type=str, default="RGAT")
    parser.add_argument("--kg_p_drop", type=float, default=0.5)
    parser.add_argument("--ui_p_drop", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.2)

    # AR-Trip
    parser.add_argument("--drifting", action="store_true")
    parser.add_argument("--guiding", action="store_true")
    parser.add_argument("--repetition_beta", type=float, default=1.0)
    parser.add_argument("--train_type", type=str, default='Penalty')
    parser.add_argument('--confidence', type=float, default=0.5)
    # ODE
    parser.add_argument("--ode", action="store_true")
    parser.add_argument("--t_unif_res", type=int, default=10,
                        help="Number of point in unfirom temporal grid used for intepolation.")
    parser.add_argument("--solver", type=str, default="dopri5", help="Name of the ODE solver (see torchdiffeq).")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for ODE solver.")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for ODE solver.")
    parser.add_argument("--dyn_hid_layers", type=int, default=3, help="Number of hidden layers in dynamics function.")
    parser.add_argument("--dyn_latent_dim", type=int, default=128, help="Hidden layer dimension in dynamics function.")
    # Model (lm).
    parser.add_argument("--lm_hid_layers", type=int, default=3, help="Number of hidden layers in intensity function.")
    parser.add_argument("--lm_latent_dim", type=int, default=128, help="Hidden layer dimension in intensity function.")
    parser.add_argument("--sig_v", type=float, default=0.6,
                        help="Observation variance.")  # Note: F 0.6 For Yelp, use 0.4

    parser.add_argument("--s_infer", action="store_true")
    parser.add_argument("--use_llm", action="store_true", help="Use LLM for training")
    parser.add_argument("--use_target_llm", action="store_true", help="Use target LLM for training")
    parser.add_argument("--use_vllm", action="store_true", help="Use vllm for training")
    parser.add_argument("--llm_embedding_dim", type=int, default=256, help="Embedding dimension for LLM")
    parser.add_argument("--use_lora", action="store_true", help="Use LLM trained with LoRA for training")
    parser.add_argument("--lora_path", type=str, default="./grpo_Yelp_f1_cat_0.75_0.1_8_lora_model/checkpoint-2250",
                        help="Path to the LoRA model")
    parser.add_argument("--lora_path2", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default="../dataset/Yelp_20250714_192438",
                        help="Path to the dataset for LLM training")
    parser.add_argument("--st_module", action="store_true", help="use SpatialTemporal module")
    # Semantic Masking parameters
    parser.add_argument("--num_semantic_parts", type=int, default=0,
                        help="Number of semantic parts for semantic-aware masking in MAE. Set to 0 to disable semantic masking and use random masking only.")
    parser.add_argument("--lambda_diversity", type=float, default=0.1, help="Weight for diversity loss in MAE.")
    parser.add_argument("--lambda_attn_reg", type=float, default=0.1,
                        help="Weight for attention regulation loss in MAE.")
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="Mask ratio for MAE.")
    parser.add_argument("--eval_dataset", type=str, default="test")
    # Yelp: ../dataset/Yelp_20250714_192438
    # Foursquare: ../dataset/travel_dataset_20250712_201017
    # Parsing command-line arguments
    args = parser.parse_args()
    args.ori_data = f'../{args.dataset_name}/home.txt'
    args.dst_data = f'../{args.dataset_name}/oot.txt'
    args.trans_data = f'../{args.dataset_name}/travel.txt'
    args.kg_path = f'../{args.dataset_name}/kg.txt'
    args.test_path = f'../{args.dataset_name}/test.txt'
    args.data_split_path = f'../{args.dataset_name}/data_split.pkl'
    args.save_path = f'../{args.dataset_name}/model_save'
    if args.dataset_name == 'Foursquare':
        args.dataset_path = '../dataset/travel_dataset_20250712_201017'
        # 强化学习后的路径
        # args.lora_path = "./sft_grpo_Foursquare_f1_cat_0.75_0.1_8_lora_model/checkpoint-1503"
        # 强化学习1轮后的路径
        # args.lora_path = "./sft_grpo_Foursquare_f1_epoch1/checkpoint-1503"
        # 只有SFT的路径
        # args.lora_path = "./sft_travel_style_lora/checkpoint-752"
        # 只有SFT1轮的路径
        args.lora_path = "./sft_travel_style_lora_Foursquare_sftepoch1/checkpoint-376"
        # 使用Refine-POI的reward
        # args.lora_path = "./sft_grpo_Foursquare_f1_RefinePOI"
        args.lora_path2 = "./sft_grpo_Foursquare_f1_RefinePOI_newlora"
    elif args.dataset_name == 'Yelp':
        args.dataset_path = '../dataset/Yelp_20250714_192438'
        # 强化学习2轮后的路径
        # args.lora_path = "./sft_grpo_Yelp_f1_cat_0.75_0.1_8_lora_model/checkpoint-2208"
        # 强化学习1轮后的路径
        # args.lora_path = "./sft_grpo_Yelp_f1_epoch1/checkpoint-2208"
        # 只有SFT的路径
        # args.lora_path = "./sft_travel_style_lora_Yelp/checkpoint-4418"
        # 只有SFT1轮的路径
        # args.lora_path = "./sft_travel_style_lora_Yelp_sftepoch1/checkpoint-553"
        # 使用了真实的f1-score
        args.lora_path = "./sft_grpo_Yelp_f1_epoch1_withRealf1"
        # 使用Refine-POI的reward
        # args.lora_path = "./sft_grpo_Yelp_f1_epoch1_RefinePOI"
        args.lora_path2 = "./sft_grpo_Yelp_f1_epoch1_RefinePOI_newlora"
    set_seeds(args.seed)
    args.name = (args.dataset_name + "_semantic" + str(args.num_semantic_parts) + "_diversity" + str(
        args.lambda_diversity)
                 + "_attnreg" + str(args.lambda_attn_reg) + "_mask" + str(args.mask_ratio))
    args.save_path = os.path.join(args.save_path, args.name)
    path_exist(args.save_path)

    # Initializing a Logger instance for recording various metrics during the training process
    # args.log_path: Path where the log file is saved
    # args.name: Name of the model, used in the log
    # args.seed: Random seed value, also recorded in the log
    # args.log: A boolean value indicating whether to output logs to the console
    logger = Logger(args.log_path, args.name, args.seed, args.log)
    logger.log(str(args))
    logger.log("Experiment name: %s" % args.name)

    # Loading the travel dataset with parameters and data paths specified in args
    data = TravelDataset(args, args.ori_data, args.dst_data, args.trans_data)

    # Checking if the knowledge graph (KG) option is enabled and loading KG data accordingly
    if args.kg:
        kg_data = KGDataset(args)
    else:
        kg_data = None
    train_data, valid_data, test_data = random_split(data, dataset_name=dataset_name, split_path=args.data_split_path)

    # train_loader = DataLoader(train_data, args.train_batch, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, args.test_batch, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, args.test_batch, shuffle=False, collate_fn=collate_fn)

    n_region = len(data.region_idx)
    max_d_length = max(len(seq) for seq in data.dsts)
    max_o_length = max(len(seq) for seq in data.oris)


    model = SPOTModel(args, len(data.poi_idx) + 1, data.region_poi, max_d_length, max_o_length,
                          d_model=args.hidden_size, n_head=4, num_encoder_layers=1, d_z=args.hidden_size,
                          kg_dataset=kg_data)
    if args.dataset_name == "Yelp" and args.eval_dataset == "test":
        test(model, os.path.join(args.save_path, "model_5.xhr"), test_loader, args, logger, n_region)
    elif args.dataset_name == "Foursquare" and args.eval_dataset == "test":
        test(model, os.path.join(args.save_path, "model_0.xhr"), test_loader, args, logger, n_region)
    elif args.dataset_name == "Yelp" and args.eval_dataset == "valid":
        test(model, os.path.join(args.save_path, "model_5.xhr"), valid_loader, args, logger, n_region)
    elif args.dataset_name == "Foursquare" and args.eval_dataset == "valid":
        test(model, os.path.join(args.save_path, "model_0.xhr"), valid_loader, args, logger, n_region)
    logger.close_log()


if __name__ == "__main__":
    main()