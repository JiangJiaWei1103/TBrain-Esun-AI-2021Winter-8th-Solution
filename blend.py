'''
Naive equal weighted blending.
Author: JiaWei Jiang

This script file is used to evaluate on the blended predicting results
of specified base models, and the blended probability distributions for
both oof and unseen data are dumped, facilitating next-level ensemble
(e.g., stacking).
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
    argparser.add_argument('--base-model-versions', type=int, nargs='+',
                           help="versions of base models to use")
    argparser.add_argument('--base-infer-versions', type=int, nargs='+',
                           help="versions of predicting results infered by "
                                "base models")
    
    args = argparser.parse_args()
    return args

def blend(exp, base_versions, data_type):
    '''Blend predicting results of base models for either oof or unseen
    dataset.
    
    Parameters:
        exp: object, a run instance to interact with wandb remote 
        base_versions: list, versions of oof or unseen predictions by
                       base models 
        datatype: str, type of the dataset, the choices are as follows:
                      {'oof', 'unseen'}
    
    Return:
        blend_result: dict, blending result
        rank_blended: pd.DataFrame, final ranking results
    '''
    n_bases = len(base_versions)
    pred_blended = None
    blend_result = {}
    
    # Configure metadata for different datatypes
    if data_type == 'oof':
        artifact_prefix = 'lgbm'
        pred_report_path = 'pred_reports/24.pkl' 
        pred_key = 'y_pred'
    elif data_type == 'unseen':
        artifact_prefix = 'lgbm_infer'
        pred_report_path = 'dt25.pkl'
        pred_key = 'y_pred_prob'
               
    # Generate blended results
    for i, v in enumerate(base_versions):
        output = exp.use_artifact(f'{artifact_prefix}:v{v}', type='output')
        output_dir = output.download()
        with open(os.path.join(output_dir, pred_report_path), 'rb') as f:
            pred_report_base = pickle.load(f)
        if i == 0:
            pred_blended = pred_report_base[pred_key] / n_bases
            blend_result['index'] = pred_report_base['index']
            if data_type == 'oof':
                blend_result['y_true'] = pred_report_base['y_true']
        else: pred_blended += pred_report_base[pred_key] / n_bases
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
    base_model_versions = args.base_model_versions
    base_infer_versions = args.base_infer_versions
    
    # Run blending 
    oof_blended, oof_rank_blended = blend(exp, base_model_versions, 'oof')
    unseen_blended, _ = blend(exp, base_infer_versions, 'unseen')

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
    output_entry = wandb.Artifact(name='lgbm', type='output')
    output_entry.add_dir("./output/")
    exp.log_artifact(output_entry)
    # =Unseen=
    print("Unseen dataset...")
    setup_local_dump('inference')
    with open(f"./output/pred_results/dt25.pkl", 'wb') as f:
        pickle.dump(unseen_blended, f)
    output_entry = wandb.Artifact(name='lgbm_infer', type='output')
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