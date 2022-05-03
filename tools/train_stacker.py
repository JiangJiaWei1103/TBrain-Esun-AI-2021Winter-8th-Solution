'''
Stacking meta-model training script.
Author: JiaWei Jiang 

This file is the training script of stacking mechanism which tries to
train next-level model based on the predicting results of the current-
level base, blending or even meta-models (i.e., restacking).
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
import yaml
import wandb

from paths import *
from metadata import *
from utils.common import load_cfg
from utils.dataset_generator import DataGenerator
from utils.common import get_artifact_info
from utils.sampler import DataSampler
from utils.xgbst_extractor import XGBstExtractor
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
    argparser.add_argument('--oof-versions', type=str, nargs='+',
                           help="versions of oof predictions infered by base,"
                                " blending or even meta-models (i.e., for "
                                "restacking)")
    argparser.add_argument('--n-folds', type=int, 
                           help="number of folds to run")
    argparser.add_argument('--eval-metrics', type=str, nargs='+', 
                           help="evaluation metrics")
    argparser.add_argument('--objective', type=str, 
                           help="objective of modeling task, the choices are "
                                "\'mcls\' or \'ranking\'")
    argparser.add_argument('--restacking', type=bool, default=False,
                           help="restacking with raw features or not")
    
    args = argparser.parse_args()
    return args

def get_meta_datasets(exp, oof_versions, objective, dg_cfg=None):
    '''Return dataset using oof predicting results as features and 
    corresponding groundtruths.
    
    Also, restacking mechanism which takes raw features into feature
    subset is implemented.
    
    Parameters:
        exp: object, a run instance to interact with wandb remote 
        oof_versions: list, versions of oof predictions by base,
                      blending or even meta-models 
        objective: str, objective of modeling task
        dg_cfg: dict, configuration for dataset generation, 
                default=None
    
    Return:
        X: pd.DataFrame, predicting results of base models
        y: ndarray, groundtruths
    '''
    X = pd.DataFrame()
    y = None

    for i, version in enumerate(oof_versions):
        af = get_artifact_info(version, 'train_eval')   # Config artifact info
        oof_df = pd.DataFrame()
        oof_pred_cols = [f'v{version}_shop_tag{s}' for s in LEG_SHOP_TAGS]
        
        output = exp.use_artifact(af, type='output')
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
        
    if dg_cfg is not None:
        print("Generating raw features for restacking...")
        # Hard coded temporarily
        dg_raw = DataGenerator(23, dg_cfg['t_window'], dg_cfg['horizon'],
                               train_leg=True, production=True, mcls=True, 
                               drop_cold_start_cli=False)
        dg_raw.run(dg_cfg['feats_to_use'])
        X_raw, y_raw = dg_raw.get_X_y()
        assert np.array_equal(y, np.array(y_raw)), "Groundtruths of oof " \
            "predictions aren't aligned with those generated from dg!!"
        X = X.join(X_raw, on='chid')
    
    return X, y

def evaluate(X, y, kf, meta_model_name, 
             cvboosters):
    '''Evaluate ranking performance using oof predicting result infered
    by best cv booster in each fold.
    
    Parameters:
        X: pd.DataFrame, predicting results of base models
        y: ndarray, groundtruths
        kf: obj, kfold cross-validator
        meta_model_name: str, meta-model to use
        cvboosters: list, best cv booster in each fold
        
    Return:
        pred_report: dict, predicting report
        prf_oof: dict, evaluation score of out of fold prediction
    '''
    oof_pk = np.zeros(len(X))
    oof_pred = np.zeros((len(X), 16))
    oof_true = np.zeros(len(y))
    
    # Get oof prediction using best cv booster in each fold
    for i, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_val = X.iloc[val_idx, :]
        if meta_model_name == 'xgb': 
            X_val = xgb.DMatrix(data=X_val)
        oof_pred_fold = cvboosters[i].predict(X_val)
        
        oof_pk[val_idx] = X.index[val_idx]
        oof_pred[val_idx, :] = oof_pred_fold
        oof_true[val_idx] = y[val_idx]
        del X_val, oof_pred_fold
    pred_report = {'index': oof_pk}
    pred_report['y_true'] = oof_true
    pred_report['y_pred'] = oof_pred
    
    # Evaluate ranking performance
    oof_pred_rank = rank_mcls_naive(pred_report['index'], oof_pred)
    evaluator_oof = EvaluatorRank("./data/raw/raw_data.parquet",
                                  t_next=24)   # Hard coded temporarily
    prf_oof = evaluator_oof.evaluate(oof_pred_rank, y)
    prf_oof = {'val_month24': prf_oof}   # Hard coded temporarily
    
    return pred_report, prf_oof 

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
        pred_report: dict, predicting report
        prf_oof: dict, evaluation score of out of fold prediction
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
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=168)
        print(f"Running cv to infer a better n_estimators...")
        if meta_model_name == 'lgbm':
            es = lgb.early_stopping(stopping_rounds=train_params['es_rounds'])
            cv_hist = lgb.cv(params=meta_model_params, 
                             train_set=train_set,
                             num_boost_round=train_params['num_iterations'],
                             folds=kf,
                             callbacks=[es],
                             return_cvbooster=True)
            print(f"=====Performance of CV=====")
            print(f"Multilogloss of val set: mean {cv_hist['multi_logloss-mean']}"
                  f" | std {cv_hist['multi_logloss-stdv']}")
            cvboosters = cv_hist['cvbooster'].boosters
            best_iter = cv_hist['cvbooster'].best_iteration
        elif meta_model_name == 'xgb':
            cvboosters = []
            cv_hist = xgb.cv(params=meta_model_params, 
                             dtrain=train_set,
                             num_boost_round=train_params['num_iterations'],
                             early_stopping_rounds=train_params['es_rounds'],
                             folds=kf,
                             callbacks=[XGBstExtractor(cvboosters)])
            print(f"=====Performance of CV=====")
            print(f"Multilogloss of val set: "
                  f"mean {cv_hist['test-mlogloss-mean'].iloc[-1]}"
                  f" | std {cv_hist['test-mlogloss-std'].iloc[-1]}")
            best_iter = cv_hist.shape[0]
        pred_report, prf_oof = evaluate(X, y, kf, meta_model_name, 
                                        cvboosters)
        
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

    return cv_hist, meta_models, pred_report, prf_oof
    
def main(args):
    '''Main function for training and evaluation process on stacker.
    
    Parameters:
        args: namespace,  
    '''
    # Setup the experiment and logger 
    exp = wandb.init(project='EsunReproduce', 
                     name='tree-based-meta',
                     job_type='train_eval_stack')
    
    # Setup basic configuration
    meta_model_name = args.meta_model_name
    oof_versions = args.oof_versions
    n_folds = args.n_folds
    eval_metrics = args.eval_metrics
    objective = args.objective
    restacking = args.restacking
    
    dg_cfg = None
    if restacking:
        # If raw features are considered in stacking process
        dg_cfg = load_cfg("./config/data_gen.yaml")
        exp.config.update({'data_gen': dg_cfg})
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
    X, y = get_meta_datasets(exp, oof_versions, objective, dg_cfg)

    # Run cross-validation
    cv_hist, meta_models, pred_report, \
    prf_oof = cv(X=X, 
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
                               f'{meta_model_name}_meta_{i}.pkl'), 'wb') as f:
            pickle.dump(meta_model, f)
    with open(f"./output/pred_reports/24.pkl", 'wb') as f:   
        # Hard coded temporarily
        pickle.dump(pred_report, f)
    if restacking:
        with open(f"./output/config/dg_cfg.yaml", 'w') as f:
            yaml.dump(dg_cfg, f)
    print("Done!!")

    # Push local storage and log performance to Wandb
    print("Start pushing storage and performance to Wandb...")
    output_entry = wandb.Artifact(name=f'{meta_model_name}_meta', 
                                  type='output')
    output_entry.add_dir("./output/")
    exp.log_artifact(output_entry)
    wandb.log(prf_oof)
    print("Done!!")
    
    print("=====Finish=====")
    wandb.finish()
    
# Main function
if __name__ == '__main__':
    args = parseargs()
    main(args)