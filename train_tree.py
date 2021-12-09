'''
NBR tree-based model training script.
Author: JiaWei Jiang 

This file is the training script of tree-based ML method trained with
raw numeric and categorical features and other hand-crafted engineered
features.

Notice that this is only the first stage of the whole work (i.e.,
binary classification for downstream ranking task).
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

from utils.common import load_cfg
from utils.dataset_generator import DataGenerator
from utils.evaluator_clf import EvaluatorCLF

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
                           help="threshold above which the observation is"
                                "classified as postive")
    argparser.add_argument('--eval-metrics', type=str, nargs='+', 
                           help="evaluation metrics")
    argparser.add_argument('--eval-like-production', type=str, default=False,
                           help="whether to evaluate model performance with"
                                "production-like primary keys; that is, all"
                                "(chid, leg_shop_tag) pairs.")
    
    args = argparser.parse_args()
    return args

def cv(dg_cfg, model_name, model_params, 
       train_params, n_folds, evaluator,
       production):
    '''Run cross-validation.
    
    Parameters:
        dg_cf: dict, configuration for dataset generation
        model_name: str, model to use
        model_params: dict, hyperdecentparameters for the specified model 
        train_params: dict hyperparameters for the training process
        n_folds: int, number of folds to run 
        evaluator: obj, evaluator for classification task
        production: bool, whether the evaluation is applied on production-
                    like dataset
        
    Return:
        clfs: list, trained model in each fold
        pred_reports: dict, predicting reports in each fold, facilitating the
               post-analysis of predicting results
        prfs: list, performance report for each fold
    '''
    clfs = []
    pred_reports = {}
    prfs = {}
    for fold in range(n_folds):
        print(f"Evaluation for fold{fold} starts...")
        t_start = proc_t()
        
        # Prepare datasets
        t_end = 24 - n_folds - dg_cfg['horizon'] + fold
        val_month = (t_end+1) + dg_cfg['horizon']
        val_month = f'val_month{val_month}'
        print("Generating training set...")
        dg_tr = DataGenerator(t_end, dg_cfg['t_window'], dg_cfg['horizon'],
                              production=False)   # train-like-production??
        dg_tr.run(dg_cfg['feats_to_use'])
        X_train, y_train = dg_tr.get_X_y()
        train_set = lgb.Dataset(data=X_train, 
                                label=y_train, 
                                categorical_feature=dg_tr.cat_features_)
        del dg_tr, X_train, y_train
        print("Done!")
        
        print("Generating validation set...")
        dg_val = DataGenerator(t_end+1, dg_cfg['t_window'], dg_cfg['horizon'],
                               production=production)
        dg_val.run(dg_cfg['feats_to_use'])
        X_val, y_val = dg_val.get_X_y()
        val_set = lgb.Dataset(data=X_val, 
                              label=y_val, 
                              reference=train_set,
                              categorical_feature=dg_val.cat_features_)
        pred_report = pd.DataFrame(index=dg_val.pk)
        del dg_val, y_val
        print("Done!\n")
        
        # Start training
        clf = lgb.train(params=model_params,
                        train_set=train_set,
                        num_boost_round=train_params['num_iterations'],
                        valid_sets=[train_set, val_set],
                        early_stopping_rounds=train_params['es_rounds'],
                        verbose_eval=train_params['verbose_eval'])
        
        # Start evaluation on validation set
#         y_tr_true = train_set.get_label() 
#         y_tr_pred = clf.predict(data=X_train, 
#                                 num_iteration=clf.best_iteration)
        print(f"Start prediction & evaluation on val set for "
              f"{val_month}...")
        y_val_true = val_set.get_label() 
        y_val_pred = clf.predict(data=X_val, 
                                 num_iteration=clf.best_iteration)
        pred_report['y_true'] = y_val_true
        pred_report['y_pred'] = y_val_pred
        prf = evaluator.evaluate(y_val_true, y_val_pred)
        print("Done!\n")
        
        # Record outputs
        clfs.append(clf)
        pred_reports[val_month] = pred_report
        prfs[val_month] = prf
        
        t_elapsed = proc_t() - t_start
        print(f"Evaluation for fold{fold} ends...")
        print(f"Total time consumption: {t_elapsed} sec.")
        
        del X_val, train_set, val_set, \
            y_val_true, y_val_pred, pred_report
    
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
    production = True if args.eval_like_production == 'True' else False 
    
    dg_cfg = load_cfg("./config/data_gen.yaml")
    model_cfg = load_cfg(f"./config/{model_name}.yaml")
    model_params = model_cfg['params']
    train_params = model_cfg['train']
    exp.config.update({'data_gen': dg_cfg,
                       'model': model_params, 
                       'train': train_params})
    evaluator = EvaluatorCLF(eval_metrics, pos_thres)
    
    # Run cross-validation
    models, pred_reports, prfs = cv(dg_cfg=dg_cfg,
                                    model_name=model_name,
                                    model_params=model_params,
                                    train_params=train_params,
                                    n_folds=n_folds,
                                    evaluator=evaluator,
                                    production=production)
    
    # Dump outputs of the experiment locally
    print("Start dumping output objects locally...")
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
    output_entry.add_dir("./output")
    exp.log_artifact(output_entry)
    wandb.log(prfs)
    print("Done!!")
    
    print("=====Finish=====")
    exp.finish()

# Main function
if __name__ == '__main__':
    args = parseargs()
    main(args)