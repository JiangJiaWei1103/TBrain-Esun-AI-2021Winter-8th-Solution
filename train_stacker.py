'''
Stacking meta-model training script.
Author: JiaWei Jiang 

This file is the training script of stacking mechanism which tries to
train next-level model based on the predicting results of the current-
level base models.
'''
# Import packages
import os 
import gc 
import argparse
import pickle

import pandas as pd 
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import wandb

from paths import *
from metadata import *
from utils.common import load_cfg
from utils.sampler import DataSampler
from utils.common import rank_mcls_naive
from utils.evaluator import EvaluatorRank
from utils.common import setup_local_dump

def parseargs():
    '''Parse and return the specified command line arguments.
    
    Parameters:
        None
        
    Return:
        args: namespace, parsed arguments
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--meta-model-name', type=str,
                           help="meta model to use")
    argparser.add_argument('--base-model-versions', type=int, nargs='+',
                           help="versions of base models to use")
    argparser.add_argument('--n-folds', type=int, 
                           help="number of folds to run")
    argparser.add_argument('--eval-metrics', type=str, nargs='+', 
                           help="evaluation metrics")
    argparser.add_argument('--objective', type=str, 
                           help="objective of modeling task, the choices are "
                                "\'mcls\' or \'ranking\'")
    
    args = argparser.parse_args()
    return args

def get_meta_datasets(exp, base_model_versions, objective):
    '''Return dataset using oof predicting results of base models as 
    features and corresponding groundtruths.
    
    Parameters:
        exp: object, a run instance to interact with wandb remote 
        base_model_versions: list, versions of well-trained base models
        objective: str, objective of modeling task
    
    Return:
        X: pd.DataFrame, predicting results of base models
        y: ndarray, groundtruths
    '''
    X = pd.DataFrame()
    y = None

    for i, version in enumerate(base_model_versions):
        oof_df = pd.DataFrame()
        oof_pred_cols = [f'v{version}_shop_tag{s}' for s in LEG_SHOP_TAGS]
        
        output = exp.use_artifact(f'lgbm:v{version}', type='output')
        output_dir = output.download()
        with open(os.path.join(output_dir, 'pred_reports/24.pkl'), 'rb') as f:
            # Hard coded temporarily
            oof_dataset = pickle.load(f)
        oof_df['chid'] = oof_dataset['index']
        oof_df['shop_tag'] = oof_dataset['y_true']
        oof_df[oof_pred_cols] = oof_dataset['y_pred']
        if i == 0:
            X = oof_df
        else: X = X.merge(oof_df, on=['chid', 'shop_tag'], how='inner')
        
        del oof_df, oof_pred_cols, output

    y = X['shop_tag']
    X.set_index('chid', drop=True, inplace=True)
    X.drop('shop_tag', axis=1, inplace=True)
    
    return X, y

def pred()
            
def cv(X, y, ds_cfg, meta_model_name, 
       meta_model_params, train_params, n_folds, objective,
       bagging=True):
    '''Run cross-validation for meta-model.
    
    Parameters:
        X: pd.DataFrame, predicting results of base models
        y: ndarray, groundtruths
        ds_cfg: dict, configuration for data sampler
        meta_model_name: str, meta-model to use
        meta_model_params: dict, hyperparameters for meta-model 
        train_params: dict, hyperparameters for the training process
        n_folds: int, number of folds to run 
        objective: str, objective of modeling task
        bagging: bool, whether to implement bagging with 5 random 
                 seeds at the final retraining phase
        
    Return:
        cv_hist: dict, evaluation history
        meta_models: list, meta models trained on whole meta X_train
                     with different random seeds
    '''
    print(f"Training and evaluation on stacker starts...")
    
    # Generate sample weights for training set in advance 
    if ds_cfg['mode'] is not None:
        print("Generating sample weights...")
        ds_tr = DataSampler(t_end=23,   # Hard coded temporarily
                            horizon=1,   # Hard coded temporarily
                            ds_cfg=ds_cfg, 
                            production=True)
        ds_tr.run()
        wt_train = ds_tr.get_weight(X.index, y)
    else: wt_train = None
    if meta_model_name == 'lgbm':
        train_set = lgb.Dataset(data=X, label=y, weight=wt_train)
    elif meta_model_name == 'xgb':
        train_set = xgb.DMatrix(data=X, label=y, weight=wt_train)
    print(f"Shape of meta X_train {X.shape} | #Clients {X.index.nunique()}")
    
    if n_folds != 1:
        # KFold for training meta-model is enabled
        print(f"Running cv to infer a better n_estimators...")
        if meta_model_name == 'lgbm':
            es = lgb.early_stopping(stopping_rounds=train_params['es_rounds'])
            cv_hist = lgb.cv(params=meta_model_params, 
                             train_set=train_set,
                             num_boost_round=train_params['num_iterations'],
                             nfold=n_folds,
                             shuffle=True,
                             seed=168,
                             callbacks=[es],
                             return_cvbooster=True)
            print(f"=====Performance of CV=====")
            print(f"Multilogloss of val set: mean {cv_hist['multi_logloss-mean']}"
                  f" | std {cv_hist['multi_logloss-stdv']}")
            best_iter = cv_hist['cvbooster'].best_iteration
        elif meta_model_name == 'xgb':
            cv_hist = xgb.cv(params=meta_model_params, 
                             dtrain=train_set,
                             num_boost_round=train_params['num_iterations'],
                             early_stopping_rounds=train_params['es_rounds'],
                             nfold=n_folds,
                             shuffle=True,
                             seed=168)
            print(f"=====Performance of CV=====")
            print(f"Multilogloss of val set: "
                  f"mean {cv_hist['test-mlogloss-mean'].iloc[-1]}"
                  f" | std {cv_hist['test-mlogloss-std'].iloc[-1]}")
            best_iter = cv_hist.shape[0]
        
    # Start training meta-models on whole training set with different random 
    # seeds
    meta_models = []
    seeds = [8, 168, 88, 888, 2022]
    best_n_rounds = int(best_iter/(1 - 1/n_folds))
    if meta_model_name == 'lgbm':
        for seed in seeds:
            meta_model_params['seed'] = seed
            meta_model = lgb.train(params=meta_model_params,
                                   train_set=train_set,
                                   num_boost_round=best_n_rounds,
                                   valid_sets=[train_set],
                                   verbose_eval=train_params['verbose_eval'])
            meta_models.append(meta_model)
    elif meta_model_name == 'xgb':
        for seed in seeds:
            meta_model_params['seed'] = seed
            meta_model = xgb.train(params=meta_model_params,
                                   dtrain=train_set,
                                   num_boost_round=best_n_rounds,
                                   evals=[(train_set, 'train')],
                                   verbose_eval=train_params['verbose_eval'])
            meta_models.append(meta_model)

    return cv_hist, meta_models
    
def main(args):
    '''Main function for training and evaluation process on stacker.
    
    Parameters:
        args: namespace,  
    '''
    # Setup the experiment and logger 
    exp = wandb.init(project='Esun', 
                     name='tree-based-meta',
                     job_type='train_eval_stack')
    
    # Setup basic configuration
    meta_model_name = args.meta_model_name
    base_model_versions = args.base_model_versions
    n_folds = args.n_folds
    eval_metrics = args.eval_metrics
    objective = args.objective
    
    ds_cfg = load_cfg("./config/data_samp.yaml")
    meta_model_cfg = load_cfg(f"./config/{meta_model_name}_meta.yaml")
    meta_model_params = meta_model_cfg['params']
    train_params = meta_model_cfg['train']
    exp.config.update({'data_samp': ds_cfg,
                       'meta_model': meta_model_params,
                       'train': train_params})
    
    # Prepare datasets
    print(f"Preparing datasets using oof predicting results from "
          "base models as features...")
    X, y = get_meta_datasets(exp, base_model_versions, objective)

    # Run cross-validation
    cv_hist, meta_models = cv(X=X, 
                              y=y, 
                              ds_cfg=ds_cfg, 
                              meta_model_name=meta_model_name, 
                              meta_model_params=meta_model_params, 
                              train_params=train_params, 
                              n_folds=n_folds, 
                              objective=objective)
    
    # Dump outputs of the experiment locally
    print("Start dumping output objects locally...")
    setup_local_dump('train_eval_stack')
    with open(f"./output/cv_hist.pkl", 'wb') as f:
        pickle.dump(cv_hist, f)
    for i, meta_model in enumerate(meta_models):
        with open(os.path.join("./output/meta_models/",
                               f'{meta_model_name}_meta_{i}.pkl', 'wb') as f:
            pickle.dump(meta_model, f)
    print("Done!!")

    # Push local storage and log performance to Wandb
    print("Start pushing storage and performance to Wandb...")
    output_entry = wandb.Artifact(name=f'{meta_model_name}_meta', 
                                  type='output')
    output_entry.add_dir("./output/")
    exp.log_artifact(output_entry)
    print("Done!!")
    
    print("=====Finish=====")
    wandb.finish()
    
# Main function
if __name__ == '__main__':
    args = parseargs()
    main(args)