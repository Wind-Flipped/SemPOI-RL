# -*- coding: utf-8 -*-
from ast import parse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
from torch.utils.data import DataLoader

import argparse
import warnings
warnings.filterwarnings('ignore')
try:
    import ipdb
except:
    pass

from utils import *
from data import TravelDataset, random_split
from model import SemPOIModel
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
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--train_batch', type=int, default=4)
    parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument('--test_batch', type=int, default=1)
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
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument("--stop_epoch", type=int, default=2) # early stopping
    parser.add_argument("--fine_stop", type=int, default=12)

    parser.add_argument("--use_llm", action="store_true", help="Use LLM for training")
    parser.add_argument("--use_target_llm", action="store_true", help="Use target LLM for training")
    parser.add_argument("--use_vllm", action="store_true", help="Use vllm for training")
    parser.add_argument("--llm_embedding_dim", type=int, default=256, help="Embedding dimension for LLM")
    parser.add_argument("--use_lora", action="store_true", help="Use LLM trained with LoRA for training")
    parser.add_argument("--lora_path", type=str, default="./grpo_travel_style_lora_model/checkpoint-5500", help="Path to the LoRA model")
    parser.add_argument("--lora_path2", type=str, default=None, help="Path to the second LoRA model. Used for reinforcement learning")
    parser.add_argument("--dataset_path", type=str, default="../dataset/Yelp_20250714_192438", help="Path to the dataset for LLM training")
    parser.add_argument("--st_module", action="store_true", help="use SpatialTemporal module")
    # Semantic Masking parameters
    parser.add_argument("--num_semantic_parts", type=int, default=0, help="Number of semantic parts for semantic-aware masking in MAE. Set to 0 to disable semantic masking and use random masking only.")
    parser.add_argument("--lambda_diversity", type=float, default=0.1, help="Weight for diversity loss in MAE.")
    parser.add_argument("--lambda_attn_reg", type=float, default=0.1, help="Weight for attention regulation loss in MAE.")
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="Mask ratio for MAE.")

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
    elif args.dataset_name == 'Yelp':
        args.dataset_path = '../dataset/Yelp_20250714_192438'
    set_seeds(args.seed)
    args.name = (args.dataset_name + "_semantic" + str(args.num_semantic_parts) + "_diversity" + str(args.lambda_diversity)
            + "_attnreg" + str(args.lambda_attn_reg) + "_mask" + str(args.mask_ratio))
    args.save_path = os.path.join(args.save_path, args.name)
    path_exist(args.save_path)

    logger = Logger(args.log_path, args.name, args.seed, args.log)
    logger.log(str(args))
    logger.log("Experiment name: %s" % args.name)

    data = TravelDataset(args, args.ori_data, args.dst_data, args.trans_data)

    train_data, valid_data, test_data = random_split(data, dataset_name=dataset_name, split_path=args.data_split_path)

    train_loader = DataLoader(train_data, args.train_batch, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, args.test_batch, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, args.test_batch, shuffle=False, collate_fn=collate_fn)

    n_region = len(data.region_idx)
    max_d_length = max(len(seq) for seq in data.dsts)
    max_o_length = max(len(seq) for seq in data.oris)

    model = SemPOIModel(args, len(data.poi_idx) + 1, data.region_poi, max_d_length, max_o_length,
                        d_model=args.hidden_size, n_head=4, num_encoder_layers=1, d_z=args.hidden_size).to(args.device)

    # Training or testing the model based on the mode specified in args

    if args.mode == 'train':
        best = train_single_phase(model, train_loader, valid_loader, test_loader, args, logger)

    elif args.mode == 'test':
        test(model, os.path.join(args.save_path, "model_best.xhr"), test_loader, args, logger, n_region)

    logger.close_log()
    
if __name__ == "__main__":
    main()