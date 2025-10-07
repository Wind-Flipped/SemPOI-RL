#!/bin/bash

num_semantic_parts=(4 8 16)
lambda_diversity_params=(0.05 0.1 0.2)
mask_ratios=(0.25 0.5 0.75)
dataset_name="Foursquare"
for mask_ratio in "${mask_ratios[@]}"; do
    for num_parts in "${num_semantic_parts[@]}"; do
        for lambda in "${lambda_diversity_params[@]}"; do
            log_file="../new_results_${dataset_name}12/${dataset_name}_mask${mask_ratio}_${num_parts}_lambda_${lambda}.log"

            mkdir -p "../new_results_${dataset_name}12"

            echo "Running experiment with mask_ratio = $mask_ratio, num_semantic_parts=$num_parts, lambda_diversity=$lambda, logging to $log_file"

            python main.py --model SPOT-Trip --dataset_name "$dataset_name" --mode train --st_module --use_llm --use_target_llm --use_vllm --hidden_size 256 --llm_embedding_dim 256 --num_semantic_parts "$num_parts" --lambda_diversity "$lambda" --mask_ratio "$mask_ratio"> "$log_file" 2>&1

            echo "Experiment with mask_ratio = $mask_ratio, num_semantic_parts=$num_parts, lambda_diversity=$lambda completed. Log saved to $log_file"
            current_time=$(date "+%Y-%m-%d %H:%M:%S")
            echo "Experiment completed at: $current_time"
            echo "----------------------------------------"
        done
    done
done
echo "Successfully completed all experiments!"