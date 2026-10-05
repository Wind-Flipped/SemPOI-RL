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

For a clean run with no existing checkpoint, use the server bundle's orchestrated
entrypoint instead of invoking the stages manually:

```bash
./start.sh train ./pipeline.env all
```

It creates and freezes the data split, trains SFT only on train UIDs, builds a
bootstrap SemPOI checkpoint for the GRPO reward, trains a UID-aligned GRPO adapter,
and finally writes each dataset's `model_best.xhr` using both adapters.

## Run Our Model
Add lora configs in `main.py` and simply run the following command to train:
```cmd
python main.py --dataset_name Foursquare --mode train --st_module --use_llm --llm_model_path "$SEMPOI_LLM_MODEL_PATH" --use_lora --lora_path "$SFT_LORA_PATH" --hidden_size 256 --llm_embedding_dim 256 --num_semantic_parts 8 --lambda_diversity 0.1 --mask_ratio 0.75
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
python eval.py --dataset_name Foursquare --st_module --use_llm --llm_model_path "$SEMPOI_LLM_MODEL_PATH" --use_lora --lora_path "$SFT_LORA_PATH" --lora_path2 "$RL_LORA_PATH" --hidden_size 256 --llm_embedding_dim 256 --num_semantic_parts 8 --lambda_diversity 0.1 --mask_ratio 0.75
```

## Export Profiles for TRIP

For the reproducible Base/SFT/RL export, fixed-city TRIP pairing, and joint
ablation runner, use `../travel_RAG/run_graduation_pipeline.sh`; the complete
handoff contract is in `../travel_RAG/graduation_pipeline/README.md`.

SemPOI-RL evaluation can export a versioned profile for each user. The profile contains the generated global travel-style text plus a diagnostic destination-conditioned category sequence and source POIs for traceability. Boundary positions are marked separately and excluded from category weights. A downstream cross-city planner should use the global style by default: the category sequence still depends on the benchmark destination, its known endpoints, and its route length, so it belongs only in a separately labelled ablation. Source POI IDs must never be treated as destination POIs.

The exported `source.destination_city` records the SemPOI-RL benchmark region used for sequence prediction. A downstream TRIP run supplies its actual planning city separately.

The exporter derives `source.model_stage` as `base`, `sft`, or `rl` from the loaded LLM/LoRA configuration and records the sequence and style checkpoint paths. Keep these fields unchanged when building paired ablations.

Checkpoint paths are explicit CLI inputs and are no longer overwritten by dataset-specific values in `eval.py`: omit `--use_lora` for Base, pass `--use_lora --lora_path "$SFT_LORA_PATH"` for SFT, and additionally pass `--lora_path2 "$RL_LORA_PATH"` for RL. Do not combine `--use_vllm` with `--use_lora`: this project does not load its PEFT adapters through the vLLM backend and now fails fast instead of silently mislabelling the run.

Run from `SemPOI-RL/code`:

```bash
python eval.py \
  --dataset_name Foursquare \
  --eval_dataset test \
  --use_llm \
  --llm_model_path "$SEMPOI_LLM_MODEL_PATH" \
  --use_lora \
  --lora_path "$SFT_LORA_PATH" \
  --lora_path2 "$RL_LORA_PATH" \
  --st_module \
  --cross_city_export_dir ../Foursquare/cross_city_profiles
```

Do not add `--use_target_llm` for a deployable or main-table experiment. That flag uses a style summarized from the held-out destination trajectory; exported profiles are marked `destination_reference`, and `travel_RAG` rejects them unless an oracle experiment is explicitly enabled.
