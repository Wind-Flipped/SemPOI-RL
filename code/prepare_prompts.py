#!/usr/bin/env python3
"""
prepare_prompts.py - Dataset preparation script

This script is responsible for:

1. Load the original tourism data

2. Use LLM to generate reference answers in batches

3. Create and save the training dataset

4. Support multi-card parallel acceleration of the generation process
"""

import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'


import argparse
import time
from datetime import datetime
from data import create_travel_text_dataset
from LLMs import TravelStyleGRPOTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Args:

    def __init__(self, dataset_name="Foursquare", device="cuda"):
        self.dataset_name = dataset_name
        self.device = device


def main():
    parser = argparse.ArgumentParser(description='Prepare for the dataset')

    parser.add_argument('--dataset_name', type=str, default='Foursquare',
                        choices=['Foursquare', 'Yelp'],
                        help='The name of the dataset (default: Foursquare)')
    parser.add_argument('--model_name', type=str, default='../LLMs/Qwen3-8B',
                        help='The path of LLM (default: ../LLMs/Qwen3-8B)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device (default: cuda)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size (default: 4)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='max number of samples (default: None means all samples)')
    parser.add_argument('--save_dataset', action='store_true', default=True,
                        help='whether to save dataset (default: True)')
    parser.add_argument('--output_name', type=str, default=None,
                        help='The name of the output dataset')

    args = parser.parse_args()

    print(f"\n{'=' * 100}")
    print(f" Start preparing the Travel style reinforcement learning dataset ")
    print(f"{'=' * 100}")
    print(f" dataset: {args.dataset_name}")
    print(f"LLM model: {args.model_name}")
    print(f" batch size: {args.batch_size}")
    print(f" maximum sample size: {args.max_samples if args.max_samples else 'all '}")
    print(f" save dataset: {args.save_dataset}")
    print(f"{'=' * 100}\n")
    start_time = time.time()

    try:
        config = Args(args.dataset_name, args.device)
        text_dataset = create_travel_text_dataset(config, args.dataset_name)

        print("🔄 Initialize the data processor..." )
        trainer = TravelStyleGRPOTrainer(
            model_name=args.model_name,
            device=args.device,
            use_lora=False,
            lora_config="./grpo_Yelp_lora_model/checkpoint-3250",
            is_train=False
        )
        print("✅ Data processor initialization completed!")

        print("🔄 Start generating the training dataset in batches...")
        dataset, reference_responses = trainer.prepare_dataset(
            text_dataset=text_dataset,
            save_dataset=args.save_dataset,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size
        )

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"\n{'=' * 100}")
        print(f" Dataset ready!")
        print(f"{'=' * 100}")
        print(f"✅ handles statistics :")
        print(f" original sample size: {len(text_dataset)}")
        print(f" Generate dataset size: {len(dataset)}")
        print(f" number of reference answers: {len(reference_responses)}")
        print(f" Processing time: {processing_time:.2f} seconds ({processing_time / 60:.1f} minutes)")
        print(f" average processing speed: {len(dataset) / processing_time:.2f} samples per second ")
        print(f" batch size: {args.batch_size}")


        if args.save_dataset:
            dataset_dir = "../dataset"
            print(f"Save directory: {dataset_dir}")


        print(f"\n{'=' * 100}")
        print(f"Successfully prepared the dataset!")
        print(f"{'=' * 100}\n")

        return dataset, reference_responses

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None



if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(" Hint: It can be run using the following command :")
        print("python prepare_prompted.py --help # view all options ")
        print("python prepare_prompted.py --max_samples 50 --batch_size 2 # test mode ")
        print("python prepare_prompted.py --dataset_name Foursquare --batch_size 4 # full pattern ")
        print()
    main()
