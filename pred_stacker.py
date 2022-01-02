'''
Stacking meta-model predicting script.
Author: JiaWei Jiang 

This file is the predicting script of stacking mechanism which tries to
do inference with the well-trained meta-model on unseen predicting 
results infered by last-level trained models.
'''
# Import packages
import os 
import gc 
import argparse
import pickle 
from time import process_time as proc_t

import pandas as pd 
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import wandb

from metadata import *
from utils.common import load_cfg
from utils.common import rank_mcls_naive
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
                           help="meta-model to use")
    argparser.add_argument('--meta-model-version', type=int,
                           help="version of the meta-model used to predict")
    argparser.add_argument('--base-model-versions', type=int, nargs='+',
                           help="versions of base models to use")
    argparser.add_argument('--base-infer-versions', type=int, nargs='+',
                           help="versions of predicting results infered by "
                                "base models")
    argparser.add_argument('--pred-month', type=int,
                           help="month to predict, please specify 25 to "
                                "enable the final production")
    argparser.add_argument('--objective', type=str, 
                           help="objective of modeling task, the choices are "
                                "\'mcls\' or \'ranking\'")
    
    args = argparser.parse_args()
    return args

def get_meta_datasets(exp, base_model_versions, base_infer_versions, 
                      objective):
    '''Return dataset using oof predicting results of base models as 
    features and corresponding groundtruths.
    
    Parameters:
        exp: object, a run instance to interact with wandb remote 
        base_model_versions: list, versions of well-trained base models
        base_infer_versions: list, versions of predicting results 
                             infered by base models 
        objective: str, objective of modeling task
    
    Return:
        X: pd.DataFrame, predicting results of base models
    '''
    X = pd.DataFrame()

    for (i, v_i), v_m in zip(enumerate(base_infer_versions), 
                             base_model_versions):
        col_names = []
        output = exp.use_artifact(f'lgbm_infer:v{v_i}', type='output')
        output_dir = output.download()
        with open(os.path.join(output_dir, 'dt25.pkl'), 'rb') as f:
            # Hard coded temporarily
            unseen_pred = pickle.load(f)
        col_names = [f'v{v_m}_shop_tag{s}' for s in LEG_SHOP_TAGS]
        X[col_names] = unseen_pred['y_pred_prob']
        if i == 0:
            X.index = unseen_pred['index']
            X.index.name = 'chid'
            
        del unseen_pred, col_names
    
    return X

def predict(X, meta_model_name, models, pred_month, 
            objective):
    '''Run inference.
    
    Parameters:
        X: pd.DataFrame, unseen predicting results infered by base 
           models
        meta_model_name: str, meta-model name
        models: list, meta-models trained with different random seeds 
        pred_month: int, month to predict
        objective: str, objective of modeling task
    
    Return:
        pred_result: pd.DataFrame, final predicting results
    '''
    print(f"Prediction for final production starts...")
    t_start = proc_t()
    
    n_folds, y_test_pred = len(models), None
    pred_result = {'index': X.index}
    if meta_model_name == 'xgb':
        X = xgb.DMatrix(data=X)
    
    # Do inference and ranking 
    print(f"Start inference on meta testing data for pred_month "
          f"{pred_month}...")
    for i, model in enumerate(models):
        if i == 0:
            y_test_pred = model.predict(data=X) / n_folds
        else: y_test_pred += model.predict(data=X) / n_folds
    if objective == 'mcls':
        # If the task is modelled as a multi-class classification
        # problem
        pred_result['y_pred_prob'] = y_test_pred
        y_test_pred = rank_mcls_naive(pred_result['index'], y_test_pred)
    pred_result['y_pred'] = y_test_pred
    print("Done!\n")
    
    t_elapsed = proc_t() - t_start
    print(f"Prediction for pred_month {pred_month} ends...")
    print(f"Total time consumption: {t_elapsed} sec.")

    return pred_result

def main(args):
    '''Main function for training and evaluation process on stacker.
    
    Parameters:
        args: namespace,  
    '''
    # Setup the experiment and logger 
    exp = wandb.init(project='Esun', 
                     name='tree-based-meta',
                     job_type='inference_stack')
    
    # Setup basic configuration
    meta_model_name = args.meta_model_name
    meta_model_version = args.meta_model_version
    base_model_versions = args.base_model_versions
    base_infer_versions =  args.base_infer_versions
    pred_month = args.pred_month
    objective = args.objective
    
    # Prepare datasets
    print(f"Preparing datasets using unseen predicting results infered"
          " by base models as features...")
    X = get_meta_datasets(exp, base_model_versions, base_infer_versions, 
                          objective)
    
    # Pull well-trained meta-model from Wandb
    if meta_model_version == 0:
        # Shortcut for accessing the latest version
        meta_model_version = 'latest'
    else: meta_model_version = f'v{meta_model_version}'
    output = exp.use_artifact(f'{meta_model_name}_meta:{meta_model_version}', 
                              type='output')
    output_dir = output.download()
    model_path = os.path.join(output_dir, 'meta_models')
    meta_models = []
    for meta_model_file in sorted(os.listdir(model_path)):
        if not meta_model_file.endswith('pkl'): continue
        with open(os.path.join(model_path, meta_model_file), 'rb') as f:
            meta_models.append(pickle.load(f))
        
    # Run inference
    pred_result = predict(X, meta_model_name, meta_models, pred_month, 
                          objective)
    
    # Dump outputs of the experiment locally
    print("Start dumping output objects locally...")
    setup_local_dump('inference')
    with open(f"./output/pred_results/dt{pred_month}.pkl", 'wb') as f:
        pickle.dump(pred_result, f)
    if objective == 'mcls':
        # Dump final ranking directly to facilitate efficient submission
        pred_result['y_pred'].to_csv("./output/submission.csv", index=False)
    print("Done!!")
    
    # Push predicting results to Wandb
    print("Start pushing predicting results to Wandb...")
    output_entry = wandb.Artifact(name=f'{meta_model_name}_infer_meta', 
                                  type='output')
    output_entry.add_dir("./output/pred_results")
    exp.log_artifact(output_entry)
    print("Done!!")
    
    print("=====Finish=====")
    exp.finish()

# Main function
if __name__ == '__main__':
    args = parseargs()
    main(args)