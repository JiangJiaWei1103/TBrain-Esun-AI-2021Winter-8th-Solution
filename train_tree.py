'''
NBR tree-based model training script.
Author: JiaWei Jiang 

This file is the training script of tree-based ML method trained with
raw numeric and categorical features and other hand-crafted engineered
features.

Notice that this file supports two modes:
1. Binary classification with self-designed ranking method
2. Multi-class classification with ranking based on output prob distrib
And now, the script focuses more on the mode 2.
'''
# Import packages
import os
import argparse
import pickle
from time import process_time as proc_t

import yaml
import pandas as pd 
import numpy as np 
import lightgbm as lgb
import wandb

from metadata import *
from utils.common import load_cfg
from utils.dataset_generator import DataGenerator
from utils.sampler import DataSampler
from utils.common import rank_mcls_naive
from utils.evaluator import EvaluatorRank   # Evaluator for ranking task 
from utils.evaluator_clf import EvaluatorCLF
from utils.common import setup_local_dump

def parseargs():
    '''Parse and return the specified command line arguments.
    
    Parameters:
        None
        
    Return:
        args: namespace, parsed arguments
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model-name', type=str, 
                           help="model to use")
    argparser.add_argument('--n-folds', type=int, 
                           help="number of folds to run")
    argparser.add_argument('--pos-thres', type=float,
                           help="threshold above which the observation is "
                                "classified as postive")
    argparser.add_argument('--eval-metrics', type=str, nargs='+', 
                           help="evaluation metrics")
    argparser.add_argument('--train-leg', type=str, default=False,
                           help="if the training set contains only samples " 
                                "with legitimate shop_tags")
    argparser.add_argument('--train-like-production', type=str, default=False,
                           help="whether to train model with all available "
                                "clients having transactions in y; that is,"
                                " #chids isn't constrained by X set.")
    argparser.add_argument('--val-like-production', type=str, default=False,
                           help="whether to make val set with all available "
                                "clients having transactions in y; that is,"
                                " #chids isn't constrained by X set.")
    argparser.add_argument('--mcls', type=str, default=True,
                           help="whether to model the task as a multi-class "
                                "classification task")
    argparser.add_argument('--eval-train-set', type=str, default=False,
                           help="whether to run evaluation on training set")
    
    args = argparser.parse_args()
    return args

def cv(dg_cfg, ds_cfg, model_name, model_params, 
       train_params, n_folds, train_leg, production_tr, 
       production_val, mcls, eval_train_set):
    '''Run cross-validation.
    
    Parameters:
        dg_cfg: dict, configuration for dataset generation
        ds_cfg: dict, configuration for data sampler
        model_name: str, model to use
        model_params: dict, hyperdecentparameters for the specified 
                      model 
        train_params: dict hyperparameters for the training process
        n_folds: int, number of folds to run 
        train_leg: bool, if the training set contains only samples with
                   legitimate shop_tags, default=False
        production_tr: bool, whether to train model with all clients
                       having txns in y
        production_val: bool, whether to make val set with all clients
                        having txns in y
        mcls: bool, whether to model task as multi-class classification
        eval_train_set: bool, whether to do evaluation on training set
            *Note: Mainly for understanding whether the model actually 
                   learns something.
        
    Return:
        clfs: list, trained model in each fold
        pred_reports: dict, predicting reports in each fold, 
                      facilitating post-analysis of predicting results
        prfs: list, performance report for each fold
    '''
    clfs = []
    pred_reports = {}
    prfs = {}
    for fold in range(n_folds):
        print(f"Evaluation for fold{fold} starts...")
        t_start = proc_t()
        
        # Prepare datasets
        t_end_tr_hard = dg_cfg['t_end_tr_hard']
        if t_end_tr_hard is None: 
            t_end_tr = 24 - n_folds - dg_cfg['horizon'] + fold 
        else: t_end_tr = t_end_tr_hard
        print("Generating training set...")
        dg_tr = DataGenerator(t_end_tr, dg_cfg['t_window'], dg_cfg['horizon'],
                              train_leg, production_tr, mcls=mcls, 
                              drop_cold_start_cli=dg_cfg['drop_cs_cli'],
                              gen_feat_tolerance=dg_cfg['gen_feat_tlrnc'],
                              drop_zero_ndcg_cli=dg_cfg['drop_0ndcg_cli'],
                              rand_samples=dg_cfg['rand_samples'])  
        dg_tr.run(dg_cfg['feats_to_use'])
        X_train, y_train = dg_tr.get_X_y()
        
        ## Generate sample weights for training set
        if ds_cfg['mode'] is not None:
            print("*Running sub-process for generating sample weights...")
            ds_tr = DataSampler(t_end_tr, dg_cfg['horizon'], ds_cfg, 
                                production_tr)
            ds_tr.run()
            wt_train = ds_tr.get_weight(dg_tr.pk.get_level_values('chid'),
                                        y_train)
        else: wt_train = None
        
        train_set = lgb.Dataset(data=X_train, 
                                label=y_train, 
                                weight=wt_train,
                                categorical_feature=dg_tr.cat_features_)
        print(f"Shape of X_train {X_train.shape} | "
              f"#Clients {dg_tr.pk.get_level_values('chid').nunique()}")
        if eval_train_set:
            pk_tr = dg_tr.pk
        else: del X_train
        del dg_tr, y_train
        
        t_end_val_hard = dg_cfg['t_end_val_hard']
        if t_end_val_hard is None: 
            t_end_val = t_end_tr + 1
        else: t_end_val = t_end_val_hard
        print("Generating validation set...")
        dg_val = DataGenerator(t_end_val, dg_cfg['t_window'], dg_cfg['horizon'],
                               train_leg, production_val, mcls=mcls,
                               drop_cold_start_cli=False,   # Hard code to False
                               gen_feat_tolerance=dg_cfg['gen_feat_tlrnc'])
        dg_val.run(dg_cfg['feats_to_use'])
        X_val, y_val = dg_val.get_X_y()
        val_set = lgb.Dataset(data=X_val, 
                              label=y_val, 
                              reference=train_set,
                              categorical_feature=dg_val.cat_features_)
        pred_report ={'index': dg_val.pk}  
        print(f"Shape of X_val {X_val.shape} | "
              f"#Clients {dg_val.pk.get_level_values('chid').nunique()}")
        del dg_val, y_val
        
        # Start training
        clf = lgb.train(params=model_params,
                        train_set=train_set,
                        num_boost_round=train_params['num_iterations'],
                        valid_sets=[train_set, val_set],
                        early_stopping_rounds=train_params['es_rounds'],
                        verbose_eval=train_params['verbose_eval'])
        
        # Start prediction on validation set (optionally on training set)
        if eval_train_set:
            train_month = t_end_tr + dg_cfg['horizon']
            print(f"Start prediction & evaluation on train set for predicting"
                  f" month {train_month}...")
            y_tr_true = train_set.get_label() 
            y_tr_pred = clf.predict(data=X_train, 
                                    num_iteration=clf.best_iteration)
        val_month = t_end_val + dg_cfg['horizon']
        print(f"Start prediction & evaluation on val set for predicting"
              f" month {val_month}...")
        y_val_true = val_set.get_label() 
        y_val_pred = clf.predict(data=X_val, 
                                 num_iteration=clf.best_iteration)
        pred_report['y_true'] = y_val_true
        pred_report['y_pred'] = y_val_pred
        if mcls:
            # If the task is modelled as a multi-class classification problem
            if eval_train_set:
                y_tr_pred = rank_mcls_naive(pk_tr, y_tr_pred)
                evaluator_tr = EvaluatorRank("./data/raw/raw_data.parquet",
                                             t_next=train_month) 
                prf_tr = evaluator_tr.evaluate(y_tr_pred, y_tr_true)
            y_val_pred = rank_mcls_naive(pred_report['index'], y_val_pred)
            evaluator_val = EvaluatorRank("./data/raw/raw_data.parquet",
                                          t_next=val_month)
            prf_val = evaluator_val.evaluate(y_val_pred, y_val_true)
        else: evaluator = EvaluatorCLF(eval_metrics, pos_thres)   # Abandoned     
        print("Done!\n")
        
        # Record outputs
        clfs.append(clf)
        pred_reports[val_month] = pred_report
        if eval_train_set:
            prfs[f'train_month{train_month}'] = prf_tr
        prfs[f'val_month{val_month}'] = prf_val
        
        t_elapsed = proc_t() - t_start
        print(f"Evaluation for fold{fold} ends...")
        print(f"Total CPU time consumption: {t_elapsed} sec.")
        
        # Free mem
        del X_val, train_set, val_set, \
            y_val_true, y_val_pred, pred_report
        if eval_train_set: del X_train, y_tr_true, y_tr_pred
    
    return clfs, pred_reports, prfs
    
def main(args):
    '''Main function for training and evaluation process.
    
    Parameters:
        args: namespace,  
    '''
    # Setup the experiment and logger
    exp = wandb.init(project='Esun',
                     name='tree-based',
                     job_type='train_eval')
    
    # Setup basic configuration
    model_name = args.model_name
    n_folds = args.n_folds
    eval_metrics = args.eval_metrics
    pos_thres = args.pos_thres
    train_leg = True if args.train_leg == 'True' else False  
    production_tr = True if args.train_like_production == 'True' else False
    production_val = True if args.val_like_production == 'True' else False 
    mcls = True if args.mcls == 'True' else False
    eval_train_set = True if args.eval_train_set == 'True' else False
    if eval_train_set:
        assert production_tr == eval_train_set, ("To evaluate on training set"
               ", training set must be processed in production scheme.")
    
    dg_cfg = load_cfg("./config/data_gen.yaml")
    ds_cfg = load_cfg("./config/data_samp.yaml")
    model_cfg = load_cfg(f"./config/{model_name}.yaml")
    model_params = model_cfg['params']
    train_params = model_cfg['train']
    exp.config.update({'data_gen': dg_cfg,
                       'data_samp': ds_cfg,
                       'model': model_params, 
                       'train': train_params})
    
    # Run cross-validation
    models, pred_reports, prfs = cv(dg_cfg=dg_cfg,
                                    ds_cfg=ds_cfg,
                                    model_name=model_name,
                                    model_params=model_params,
                                    train_params=train_params,
                                    n_folds=n_folds,
                                    train_leg=train_leg,
                                    production_tr=production_tr,
                                    production_val=production_val,
                                    mcls=mcls,
                                    eval_train_set=eval_train_set)
    
    # Dump outputs of the experiment locally
    print("Start dumping output objects locally...")
    setup_local_dump('train_eval')
    for (val_month, pred_report), clf in zip(pred_reports.items(), models):
        with open(f"./output/models/{val_month}.pkl", 'wb') as f:
            pickle.dump(clf, f)
        with open(f"./output/pred_reports/{val_month}.pkl", 'wb') as f:
            pickle.dump(pred_report, f)
    print("Done!!")
    
    # Push local storage and log performance to Wandb
    # Note: Name of the Artifact must be unique across a project; that is, we
    #       can't use the same name for different type specification
    print("Start pushing storage and performance to Wandb...")
    output_entry = wandb.Artifact(name=model_name, 
                                  type='output')
    output_entry.add_dir("./output/")
    exp.log_artifact(output_entry)
    wandb.log(prfs)
    print("Done!!")
    
    print("=====Finish=====")
    exp.finish()

# Main function
if __name__ == '__main__':
    args = parseargs()
    main(args)