# SPOT-Trip

This is the official implementation of our paper titled **"SPOT-Trip: Dual-Preference Driven Out-of-Town Trip Recommendation"**.

Pytorch versions are provided.

> Pytorch: https://pytorch.org

## Data

We have released the travel behavior dataset Foursquare and Yelp which are generated based on the [Foursquare](https://sites.google.com/site/yangdingqi/home/foursquaredataset) and [Yelp](https://www.yelp.com.tw/dataset) dataset. You can run the model with these out-of-town data provided in the respective folder.


## Run Our Model

Simply run the following command to train and evaluate:
```cmd
cd ./code
python main.py --ori_data {...} --dst_data {...} --trans_data {...} --save_path {...} --model SPOT-Trip --mode train --kg --train_trans --ode --s_infer
```


## Prepare Text Data
If you want to use text data, you need to prepare the text data first. You can run the following command to prepare the text data:
```cmd
cd ./code
python prepare_prompts.py --dataset_name Foursquare --batch_size 4
```

## Train
To train the model, you can run the following command:
```cmd
cd ./code
python main.py --model SPOT-Trip --mode train --train_trans --ode --s_infer --use_llm
```

## Train with target LLM
If you want to train the model with a specific LLM, you can run the following command:
```cmd
cd ./code
python main.py --model SPOT-Trip --mode train --train_trans --ode --s_infer --use_llm --use_target_llm
```

## RL for Training
If you want to train the model with RL, you can run the following command:
```cmd
cd ./code
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python LLMs.py
```

## Evaluate with LLM trained with RL
If you want to evaluate the model with LLM trained with RL, you can run the following command:
```cmd
cd ./code
python main.py --model SPOT-Trip --mode train --train_trans --ode --s_infer --use_llm --use_vllm --use_lora --llm_embedding_dim 256
```

## Use SpatialTemporal Module for training
If you want to use the SpatialTemporal module instead of ODE for training, you can run the following command:
```cmd
cd ./code
python main.py --model SPOT-Trip --mode train --train_trans --s_infer --use_llm --use_target_llm --use_vllm --st_module --llm_embedding_dim 256
```