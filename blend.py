'''
Base or meta-model blender.
Author: JiaWei Jiang

This script file is used to evaluate on the blended predicting results
of specified base or meta-models, and the blended probability distribs
for both oof and unseen data are dumped, facilitating next-level 
ensemble (e.g., stacking).
'''
# Import packages 
import os 
import pickle
import argparse

import pandas as pd 
import numpy as np 
import wandb 

from paths import *
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
    argparser.add_argument('--oof-versions', type=str, nargs='+',
                           help="versions of oof predictions infered by base,"
                                " blending or even meta-models (i.e., for "
                                "restacking)")
    argparser.add_argument('--unseen-versions', type=str, nargs='+',
                           help="versions of unseen predictions infered by "
                                "base, blending or even meta-models (i.e., for"
                                " restacking)")
    argparser.add_argument('--meta', type=bool, default=False,
                           help="if true, then models to blend are all meta-"
                                "models")
    argparser.add_argument('--weights', type=float, nargs='+', default=None,
                           help="weight for each model to blend")
    
    args = argparser.parse_args()
    return args

def blend(exp, versions, data_type, meta=False,
          wts=None):
    '''Blend predicting results of base models for either oof or unseen
    dataset.
    
    Parameters:
        exp: object, a run instance to interact with wandb remote 
        versions: list, versions of oof or unseen predictions by base or
                  meta-models 
        datatype: str, type of the dataset, the choices are as follows:
                      {'oof', 'unseen'}
        meta: bool, whether models to blend are meta-models
        wts: list, weight vector used to blend the specified models, 
             default=None
    
    Return:
        blend_result: dict, blending result
        rank_blended: pd.DataFrame, final ranking results
    '''
    n_models = len(versions)
    if wts is None:
        # Blending with equal weights is used
        wts = [1/n_models for _ in range(n_models)]
    pred_blended = None
    blend_result = {}
    
    # Configure metadata for different datatypes
    if data_type == 'oof':
        job_type = ''
        pred_report_path = 'pred_reports/24.pkl' 
        pred_key = 'y_pred'
    elif data_type == 'unseen':
        job_type = '_infer'
        pred_report_path = 'dt25.pkl'
        pred_key = 'y_pred_prob'
    
    # Configure model type 
    model_type = '_meta' if meta else ''
               
    # Generate blended results
    for i, v in enumerate(versions):
        if v.startswith('l'):
            model_name = 'lgbm'
        elif v.startswith('x'):
            model_name = 'xgb'
        elif v.startswith('b'):
            model_name = 'blend'
        v = int(''.join([c for c in v if c.isdigit()]))
        output = exp.use_artifact(f'{model_name}{job_type}{model_type}:v{v}', 
                                  type='output')
        output_dir = output.download()
        with open(os.path.join(output_dir, pred_report_path), 'rb') as f:
            pred_report_base = pickle.load(f)
        if i == 0:
            pred_blended = pred_report_base[pred_key] * wts[i]
            blend_result['index'] = pred_report_base['index']
            if data_type == 'oof':
                blend_result['y_true'] = pred_report_base['y_true']
        else: pred_blended += pred_report_base[pred_key] * wts[i]
        del output, pred_report_base
    blend_result[pred_key] = pred_blended
    
    # Final ranking 
    rank_blended = rank_mcls_naive(blend_result['index'], pred_blended)
    if data_type == 'unseen': blend_result['y_pred'] = rank_blended
            
    return blend_result, rank_blended

def main(args):
    '''Main function for blending the predicting results of different 
    base models.
    
    Parameters:
        args: namespace, parsed arguments
    '''
    # Setup experiment and logger
    exp = wandb.init(project='Esun',
                     name='tree-based-blend',
                     job_type='blend')
    
    # Setup basic configuration 
    oof_versions = args.oof_versions
    unseen_versions = args.unseen_versions
    meta = args.meta
    wts = args.weights
    
    # Run blending 
    print(meta)
    oof_blended, oof_rank_blended = blend(exp, oof_versions, 'oof', meta, 
                                          wts)
    unseen_blended, _ = blend(exp, unseen_versions, 'unseen', meta, 
                              wts)

    # Run evaluation on blended oof prediction
    evaluator = EvaluatorRank(DATA_PATH_RAW, t_next=24)
    prf = evaluator.evaluate(oof_rank_blended)
    prf = {'val_month24': prf}
    
    # Dump outputs of the experiment locally
    print("Start dumping output objects locally and pushing storage to Wandb "
          "for oof and unseen datasets, separately...")
    # =OOF=
    print("OOF dataset...")
    setup_local_dump('train_eval')
    with open(f"./output/pred_reports/24.pkl", 'wb') as f:
        pickle.dump(oof_blended, f)
    output_entry = wandb.Artifact(name='blend', type='output')
    output_entry.add_dir("./output/")
    exp.log_artifact(output_entry)
    # =Unseen=
    print("Unseen dataset...")
    setup_local_dump('inference')
    with open(f"./output/dt25.pkl", 'wb') as f:
        pickle.dump(unseen_blended, f)
    output_entry = wandb.Artifact(name='blend_infer', type='output')
    output_entry.add_dir("./output/")
    exp.log_artifact(output_entry)
    print("Done!!")
    
    # Push blended results to Wandb
    print("Start pushing performance to Wandb...")
    wandb.log(prf)
    print("Done!!")
    
    print("=====Finish=====")
    exp.finish()

if __name__ == '__main__':
    args = parseargs()
    main(args)