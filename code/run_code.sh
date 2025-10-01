#!/bin/bash

num_semantic_parts=(0)
lambda_diversity_params=(0.1)
mask_ratios=(0.25 0.5 0.75)
#dataset_name="Foursquare"
#
#for mask_ratio in "${mask_ratios[@]}"; do
#  # 遍历num_semantic_parts数组
#    for num_parts in "${num_semantic_parts[@]}"; do
#        # 遍历lambda_diversity数组
#        for lambda in "${lambda_diversity_params[@]}"; do
#            # 构造日志文件路径（包含两个参数值）
#            log_file="../new_results_${dataset_name}11/${dataset_name}_mask${mask_ratio}_${num_parts}_lambda_${lambda}.log"
#
#            # 确保结果目录存在
#            mkdir -p "../new_results_${dataset_name}11"
#
#            # 打印当前运行的信息（可选）
#            echo "Running experiment with mask_ratio = $mask_ratio, num_semantic_parts=$num_parts, lambda_diversity=$lambda, logging to $log_file"
#
#            # 运行Python命令，传递两个参数，并重定向输出到日志文件
#            python main.py --model SPOT-Trip --dataset_name "$dataset_name" --mode train --train_trans --st_module --use_llm --use_target_llm --use_vllm --hidden_size 256 --llm_embedding_dim 256 --num_semantic_parts "$num_parts" --lambda_diversity "$lambda" --mask_ratio "$mask_ratio"> "$log_file" 2>&1
#
#            # 可选：打印完成信息
#            echo "Experiment with mask_ratio = $mask_ratio, num_semantic_parts=$num_parts, lambda_diversity=$lambda completed. Log saved to $log_file"
#            current_time=$(date "+%Y-%m-%d %H:%M:%S")
#            echo "Experiment completed at: $current_time"
#            echo "----------------------------------------"
#        done
#    done
#done


dataset_name="Foursquare"
for mask_ratio in "${mask_ratios[@]}"; do
  # 遍历num_semantic_parts数组
    for num_parts in "${num_semantic_parts[@]}"; do
        # 遍历lambda_diversity数组
        for lambda in "${lambda_diversity_params[@]}"; do
            # 构造日志文件路径（包含两个参数值）
            log_file="../new_results_${dataset_name}12/${dataset_name}_mask${mask_ratio}_${num_parts}_lambda_${lambda}.log"

            # 确保结果目录存在
            mkdir -p "../new_results_${dataset_name}12"

            # 打印当前运行的信息（可选）
            echo "Running experiment with mask_ratio = $mask_ratio, num_semantic_parts=$num_parts, lambda_diversity=$lambda, logging to $log_file"

            # 运行Python命令，传递两个参数，并重定向输出到日志文件
            python main.py --model SPOT-Trip --dataset_name "$dataset_name" --mode train --train_trans --st_module --use_llm --use_target_llm --use_vllm --hidden_size 256 --llm_embedding_dim 256 --num_semantic_parts "$num_parts" --lambda_diversity "$lambda" --mask_ratio "$mask_ratio"> "$log_file" 2>&1

            # 可选：打印完成信息
            echo "Experiment with mask_ratio = $mask_ratio, num_semantic_parts=$num_parts, lambda_diversity=$lambda completed. Log saved to $log_file"
            current_time=$(date "+%Y-%m-%d %H:%M:%S")
            echo "Experiment completed at: $current_time"
            echo "----------------------------------------"
        done
    done
done
echo "所有命令已执行完成，结果已保存到相应的log文件中。"