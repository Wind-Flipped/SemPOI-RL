# SemPOI-RL

This is the implementation of our paper titled **SemPOI-RL: Reinforcement Learning with Semantic Alignment for Cross-City POI Recommendation**.


## Data

We have released the travel behavior dataset Foursquare and Yelp which are generated based on the [Foursquare](https://sites.google.com/site/yangdingqi/home/foursquaredataset) and [Yelp](https://www.yelp.com.tw/dataset) dataset. You can run the model with these out-of-town data provided in the respective folder. The data is the same as [SPOT-Trip](https://github.com/Yinghui-Liu/SPOT-Trip/tree/main) model.

## Prepare Text Data
If you want to use text data, you need to prepare the text data first. You can run the following command to prepare the text data:
```cmd
cd ./code
python prepare_prompts.py --dataset_name Foursquare --batch_size 4
```
Or you can directly use our provided text data in the `./dataset` folder.

## Download the LLM
We use the [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) as our LLM model and [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) as our text embedding model. You can download the model from Hugging Face. Make sure you have access to the model. After downloading, place the model in the `./LLMs` folder.


## SFT
We provide the code for SFT training. You can change your dataset directory in `rl.py`. Run the following command to train the SFT model:
```cmd
python rl.py --sft --dataset_name Foursquare --sft_run_name Foursquare_SFT_v1
```

## Run Our Model
Add lora configs in `main.py` and simply run the following command to train:
```cmd
python main.py --dataset_name Foursquare --mode train --st_module --use_llm --use_lora --hidden_size 256 --llm_embedding_dim 256 --num_semantic_parts 8 --lambda_diversity 0.1 --mask_ratio 0.75
```

## RL for Training
If you want to train the model with RL, you can run the following command:
```cmd
python rl.py --dataset_name Foursquare --lora_config "$your_sft_lora_config"
```


## Run the whole model for final results
If you want to run the whole model for final results, open "./code/eval.py" file add modify the lora_path and model_*.xhr.
Run the following command:
```cmd
cd ./code
python eval.py --dataset_name Foursquare --st_module --use_llm --use_lora --lora_path "$sft_lora_path" --lora_path2 "$rl_lora_path" --hidden_size 256 --llm_embedding_dim 256 --num_semantic_parts 8 --lambda_diversity 0.1 --mask_ratio 0.75 
```