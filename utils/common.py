'''
Commonly used utility functions.
Author: JiaWei Jiang

This file includes some commonly used utility functions in modeling
process.
'''
# Import packages 
import os 
import shutil

import yaml
import pandas as pd 
import numpy as np

from metadata import *

# Utility function definitions
def load_cfg(cfg_path):
    '''Load and return the specified configuration.
    
    Parameters:
        cfg_path: str, path of the configuration file
    
    Return:
        cfg: dict, configuration 
    '''
    with open(cfg_path, 'r') as f:
        cfg = yaml.full_load(f)
    return cfg

def rank_mcls_naive(pk, y_val_rank):
    '''Naive ranking based on the probability distribution of multi-
    class classifier.
    
    Parameters:
        pk: pd.DataFrame.index, client identifiers
            *Note: To model the task as a multi-class classification problem, 
                   pk contains duplicated `chid`s corresponding to different
                   `shop_tag`s needed to be dropped.
        y_val_rank: np.array, predicting probability distribution
    
    Return:
        final_ranks: pd.DataFrame, final ranking results
    '''
    shop_tags_sorted = np.fliplr(np.argsort(y_val_rank))
    shop_tags_top3 = shop_tags_sorted[:, :3]
    
    # Generat final ranking report
    final_ranks = pd.DataFrame(shop_tags_top3, index=pk)
    final_ranks.reset_index(drop=False, inplace=True)
    final_ranks.columns = ['chid', 'top1', 'top2', 'top3']
    final_ranks.drop_duplicates(subset=['chid'], 
                                inplace=True, 
                                ignore_index=True)
    final_ranks.replace({k: v for k, v in enumerate(LEG_SHOP_TAGS)},
                        inplace=True)   # Map shop_tag indices back to 
                                        # original tag number
    
    return final_ranks

def setup_local_dump(task):
    '''Setup local dumping buffer for process outputs.
    
    Parameters:
        task: str, task type of the process
        
    Return:
        None
    '''
    dump_path = "./output"
    if os.path.exists(dump_path):
        shutil.rmtree(dump_path)
    
    os.mkdir(dump_path)
    if task == 'train_eval':
        os.mkdir(os.path.join(dump_path, 'config'))
        os.mkdir(os.path.join(dump_path, 'models'))
        os.mkdir(os.path.join(dump_path, 'pred_reports'))
    elif task == 'inference':
        os.mkdir(os.path.join(dump_path, 'pred_results'))
    elif task == 'train_eval_stack':
        os.mkdir(os.path.join(dump_path, 'config'))
        os.mkdir(os.path.join(dump_path, 'meta_models'))
        os.mkdir(os.path.join(dump_path, 'pred_reports'))

def get_artifact_info(version, job_type):
    '''Return reconstructed artifact name and version given abbrev. of 
    it from CLI argument.
    
    Parameters:
        version: str, abbreviation of artifact information specified in 
                 CLI argument
        job_type: str, type of the running task
                 
    Return:
        artifact_info: str, reconstructed artifact information
    '''
    if version.startswith('l'):
        model_name = 'lgbm'
    elif version.startswith('x'):
        model_name = 'xgb'
    elif version.startswith('b'):
        model_name = 'blend'
    meta = '_meta' if version[1] == 'm' else ''
    job_type = '_infer' if job_type == 'infer' else ''
    
    v = int(''.join([c for c in version if c.isdigit()]))
    
    artifact_info = f'{model_name}{job_type}{meta}:v{v}'
    
    return artifact_info
    