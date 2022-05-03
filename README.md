# 玉山人工智慧公開挑戰賽2021冬季賽 - 信用卡消費類別推薦
###### tags: `Side Project or Contest`

> 1. The complete workflow is shared at [Competition | 玉山人工智慧公開挑戰賽2021冬季賽 — 信用卡消費類別推薦8th Solution](https://medium.com/@waynechuang97/competition-%E7%8E%89%E5%B1%B1%E4%BA%BA%E5%B7%A5%E6%99%BA%E6%85%A7%E5%85%AC%E9%96%8B%E6%8C%91%E6%88%B0%E8%B3%BD2021%E5%86%AC%E5%AD%A3%E8%B3%BD-%E4%BF%A1%E7%94%A8%E5%8D%A1%E6%B6%88%E8%B2%BB%E9%A1%9E%E5%88%A5%E6%8E%A8%E8%96%A68th-solution-21ab52ef7dd4).
> 2. The reproduced result is logged at [Wandb | EsunReproduce](https://wandb.ai/jiangjiawei1103/EsunReproduce?workspace=user-jiangjiawei1103)

## Objective
Given raw features of customers (*e.g.*, customer properties, transaction information of credit cards), the objective is to predict and output (recommend) the **ordering of consumption categories (*i.e.*, `shop_tag`)** with **total amount of consumption** (*i.e.*, `txn_amt`) ranked top 3. The illustration is as follows:
| chid     | top1 | top2 | top3 |
| -------- | ---- | ---- | ---- |
| 10128239 | 18   | 10   | 6    |
| 10077943 | 48   | 22   | 6    |

## Prerequisite
First, 49 consumption categories (*i.e.*, `shop_tag`) are given, but participants are asked to predict only 16 of them; that is, only these 16 categories can appear in the final submision (recommendation). Second, all the 500000 customer (*i.e.*, `chid`) are the predicting targets; in other words, all of them should be included in the final submission.

## How to Run
Following is the step-by-step guideline for generating the final result. For quicker inference for the best result, please skip this section and go to [Quick Inference for The Best Result](#Quick-Inference-for-The-Best-Result) directly.
### *a. Data Preparation*
The very first step is to generate the raw data (*e.g.*, raw DataFrame, feature maps) for further EDA and feature engineering process. With high memory consumption, raw data is generated as follows (following argument setting is just an example):
#### 1. Convert `dtype` and Partition
a. Put raw data `tbrain_cc_training_48tags_hash_final.csv` in folder `data/raw/`<br>
b. Run command 
```
python -m data_preparation.convert_type
```
> Output partitioned files are dumped under path `data/partitioned/`.
#### 2. Generate Raw DataFrame 
Run command 
```
python -m data_preparation.gen_raw_df
```
> Output raw DataFrames `raw_data.parquet` and `raw_txn_amts.parquet` are dumped under path `data/raw/`.
#### 3. Generate Feature Map 
Run command 
```
python -m data_preparation.gen_feat_map --feat-type <feat-type>
```
> Output feature maps are dumped under either `data/processed/feat_map/` or `data/processed/feat_map_txn_amt/`.
#### 4. Generate Purchasing Map 
Run command 
```
python -m data_preparation.gen_purch_map
```
> Output purchasing maps `purch_maps.pkl` is dumped under path `data/processed/`.

### *b. Base Model Training*
Complete training process is configured by three configuration files, `config/data_gen.yaml`, `config/data_samp.yaml` and `config/lgbm.yaml`.
`data_gen.yaml` controls data constraint, feature engineering and final dataset generation. `data_samp.yaml` is related to sample weight generation, and `lgbm.yaml` is the hyperparameter setting for LightGBM classifier.
To better manage experimental trials, I use `Wandb` to record training process, log debugging message, and store output objects. 
Base model is trained as follows (following argument setting is just an example):
#### 1. Configure `config/data_gen.yaml`
For more detailed information, please refer to [`data_gen_template.yaml`](https://github.com/JiangJiaWei1103/TBrain-Esun-AI-2021Winter/blob/master/config/data_gen_template.yaml).
#### 2. Configure `config/data_samp.yaml`
Default setting can obtain the best performance. Please feel free to play around with it.
#### 3. Configure `config/lgbm.yaml`
Default setting can obtain relatively stable performance. And, this is the hyperparameter set I use to train all the base models. If there's no GPU support, please set `device` option to `cpu`.
#### 4. Train Base Model
Run command 
```
python -m tools.train_tree --model-name lgbm --n-folds 1 --eval-metrics ndcg@3 --train-leg True --train-like-production True --val-like-production True --mcls True --eval-train-set True
```
For more detailed information about arguments, please run command `python -m tools.train_tree -h`<br>
Output structure is as follows:
```
    output/
       ├── config/ 
       ├── models/
       ├── pred_reports/
```
> All dumped objects are pushed to `Wandb` remote.

### *c. Base Model Inference*
For single base model inference, pre-trained LightGBM classifier is pulled from `Wandb` remote first, then the probability distribution is predicted. 
Single base model inference is run as follows (following argument setting is just an example):<br>
Run command 
```
python -m tools.pred_tree --model-name lgbm --model-version 0 --val-month 24 --pred-month 25 --mcls True
```
For more detailed information about arguments, please run command `python -m tools.pred_tree -h`<br>
Output structure is as follows:
```
    output/
       ├── pred_results/
       ├── submission.csv   # For quick submission
```
> All dumped objects, excluding `outputs/submission.csv`, are pushed to `Wandb` remote.

### *d. Stacker Training*
Because single model faces performance bottleneck, so **stacking mechanism** is implemented to boost the performance. 
Stacker training process is controlled by `config/lgbm_meta.yaml` or `config/xgb_meta.yaml` depending on stacker choice. Further more, if **restacking** (*i.e.*, stacking with other raw features) is enabled, then setting `config/data_gen.yaml` is necessary.
Stacker is trained as follows (following argument setting is just an example):
#### 1. (Optional, depending on restacking or not) Configure `config/data_gen.yaml` 
For more detailed information, please refer to [`data_gen_template.yaml`](https://github.com/JiangJiaWei1103/TBrain-Esun-AI-2021Winter/blob/master/config/data_gen_template.yaml).
#### 2. Train stacker
Run command 
```
python -m tools.train_stacker --meta-model-name xgb --n-folds 5 --eval-metrics ndcg@3 --objective mcls --oof-versions l184 l186 l187 l190 l192 l194 l195 b1 b2 b3
```
For more detailed information about arguments, please run command `python -m tools.train_stacker -h`<br>
Output structure is as follows:
```
    output/
       ├── cv_hist.pkl
       ├── meta_models/
       ├── pred_reports/
       ├── config/
```
> All dumped objects are pushed to `Wandb` remote.

### *e. Stacker Inference*
For meta model inference, pre-trained LightGBM or XGB stacker (*i.e.*, classifier) is pulled from `Wandb` remote first, then the probability distribution is predicted. 
Meta model inference is run as follows (following argument setting is just an example):<br>
Run command 
```
python -m tools.pred_stacker --meta-model-name xgb --meta-model-version 0 --pred-month 25 --objective mcls --oof-versions l184 l186 l187 l190 l192 l194 l195 b1 b2 b3 --unseen-versions l48 l50 l51 l54 l58 l60 l61 b1 b2 b3
```
For more detailed information about arguments, please run command `python -m tools.pred_stacker -h`<br>
Output structure is as follows:
```
    output/
       ├── pred_results/
       ├── submission.csv   # For quick submission
```
> All dumped objects, excluding `outputs/submission.csv`, are pushed to `Wandb` remote.

### *f. Blending with Bayesian Optimization*
To better combine merits of different models (either base models or meta models), blending with coefficients optimized by **Bayesian optimization** is implemented.
Blending is run as follows (following argument setting is just an example):
#### 1. Derive Blending Coefficients
Run Bayesian optimization in [`ensemble.ipynb`](https://github.com/JiangJiaWei1103/TBrain-Esun-AI-2021Winter/blob/master/supplementary/ensemble.ipynb) and obtain blending coefficients.
#### 2. Blend Probability Distributions Infered by Different Models
Run command 
```
python -m tools.blend --oof-versions l16 l18 x8 x10 --unseen-versions l10 l12 x7 x9 --weights 0.144372 0.856641 0.307942 0.19094 --meta True
```
For more detailed information about arguments, please run command `python -m tools.blend -h`<br>
Output structure is as follows:
```
1. For blending oof predictions:
    output/
        ├── pred_reports/
        
2. For blending unseen predictions:
    output/
        ├── pred_results/
        ├── submission.csv
```
> All dumped objects are pushed to `Wandb` remote.

## Quick Inference for The Best Result
This section provides the shortcut to obtain the performance on leaderboard. The best result can be generated as follows (following argument setting is just an example):<br>
#### 1. Modify `Wandb` Project Name
Modify `project` parameter in `wandb.init()` to `Esun` in script [`blend.py`](https://github.com/JiangJiaWei1103/TBrain-Esun-AI-2021Winter-8th-Solution/blob/master/tools/blend.py).
#### 2. Blend Probability Distributions Infered by Different Models
Run command 
```
python -m tools.blend --oof-versions l16 l18 x8 x10 --unseen-versions l10 l12 x7 x9 --weights 0.144372 0.856641 0.307942 0.19094 --meta True
```