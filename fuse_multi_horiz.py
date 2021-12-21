'''
Multi-horizon model fusion (weighted ensembling) script.
Author: JiaWei Jiang

This file is the ensembling script of predicting results inferenced by
models trained on different predicting horizons. 

The fusion is done with the granularity of `shop_tag`; that is, there's
different weight combination for each `shop_tag` to align with its avg
transaction gap (frequency) characteristics (e.g., `shop_tag`37 shows
relatively high frequency.).
'''
# Import packages
import os 
import argparse
import pickle 

import pandas as pd 
import numpy as np
import wandb 

from metadata import *
from fe import get_txn_related_feat
from utils.common import rank_mcls_naive
from utils.common import setup_local_dump

# Variable definitions 
GAP_BREAK_PTS = [_ for _ in range(1, 13)] # 18

def parseargs():
    '''Parse and return the specified command line arguments.
    
    Parameters:
        None
        
    Return:
        args: namespace, parsed arguments
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model-names', type=str, nargs='+',
                           help="models to use")
    argparser.add_argument('--infer-versions', type=int, nargs='+',
                           help="versions of predicting results used "
                                "to fuse (ensemble), short horizon first")
    argparser.add_argument('--avg-gap-break-pts', type=int, nargs='+',
                           help="break points of average transaction gaps "
                                "used to determine weights of fusion")
    
    args = argparser.parse_args()
    return args

def fuse(pred_results, avg_gap_break_pts, t_end=24):
    '''Fuse predicting results infered by multi-horizon models based on
    derived ensemble weights.
    
    Parameters:
        pred_results: dict, predicting results infered by multi-horizon 
                      models 
        avg_gap_break_pts: list, break points of avg txn gaps used to 
                           derive ensemble weights
    
    Return:
        None
    '''
    print(f"Fusion of predicting results of multi-horizon models starts...")
        
    # Derive ensemble weights
    print(f"Deriving ensemble weights...")
    chid_ratios = get_chid_ratios(t_end)
    wts = {}
    for shop_tag in LEG_SHOP_TAGS:
        wt_vec = [0]
        chid_ratios_shop_tag = chid_ratios[shop_tag]
        for avg_gap_bpt in avg_gap_break_pts:
            wt_vec.append(chid_ratios_shop_tag[avg_gap_bpt-1])
        wt_vec.append(1)
        wt_vec = np.diff(np.array(wt_vec))
        wts[shop_tag] = wt_vec
    wt_mat = np.array(list(wts.values()))
    
    # Fuse predicting results using derived ensemble weights
    print(f"Fusing probability distributions...")
    pred_result_ens = {}
    pred_prob_fuse = None#np.zeros((N_CLIENTS, len(LEG_SHOP_TAGS)))
    for i, (_, pred_result) in enumerate(pred_results.items()):
        if i == 0: 
            pred_result_ens['index'] = pred_result['index']
            pred_prob_fuse = (pred_result['y_pred_prob'] * wt_mat[:, i])
        pred_prob_fuse += (pred_result['y_pred_prob'] * wt_mat[:, i])
    pred_result_ens['y_pred_prob_fuse'] = pred_prob_fuse
    
    # Rank based on fused predicting probability distribution
    print(f"Ranking for final production...")
    y_pred_fuse = rank_mcls_naive(pred_result_ens['index'], pred_prob_fuse)
    pred_result_ens['y_pred_fuse'] = y_pred_fuse
    print("Done!!\n")
    
    return pred_result_ens

def get_chid_ratios(t_end):
    '''Return accumulative ratio of #chids within each avg txn gap 
    interval for each shop_tag.
    
    Parameters:
        None
    
    Return:
        chid_ratios: dict, accumulative ratio of #chids
    '''
    txn_gap_vecs = get_txn_related_feat(t_end, 'avg_gap', leg_only=True)
    txn_gap_vecs = np.array(list(txn_gap_vecs.values()))
    
    # Get ratio of #chids within each gap interval
    chid_ratios = {}
    for shop_tag, idx in LEG_SHOP_TAG_MAP.items():
        txn_gaps_shop_tag = txn_gap_vecs[:, idx]
        txn_gaps_shop_tag = txn_gaps_shop_tag[txn_gaps_shop_tag != 100]

        n_chids_binned = []
        for upper_bound in GAP_BREAK_PTS:
            gap_intv = ((txn_gaps_shop_tag >= upper_bound-1) & 
                        (txn_gaps_shop_tag < upper_bound))
            n_chids_intv = np.sum(np.where(gap_intv, True, False))
            n_chids_binned.append(n_chids_intv)
        n_chids_binned.append(len(txn_gaps_shop_tag) - np.sum(n_chids_binned))
        chid_ratios[shop_tag] = (np.cumsum(np.array(n_chids_binned) / 
                                           np.sum(n_chids_binned)))
    
    return chid_ratios
                              
def main(args):
    '''Main function for fusing predictions infered by multi-horizon 
    models. 
    
    Parameters:
        args: namespace, 
    '''
    # Setup the experiment and logger
    exp = wandb.init(project='Esun',
                     name='tree-based',
                     job_type='multi_horiz_fusion')
    
    # Setup basic configuration
    model_names = args.model_names
    infer_versions = args.infer_versions
    avg_gap_break_pts = args.avg_gap_break_pts
    
    # Pull predicting results from Wandb
    pred_results = {}
    for model_name, infer_version in zip(model_names, infer_versions):
        output = exp.use_artifact(f'{model_name}_infer:v{infer_version}', 
                                  type='output')
        output_dir = output.download()
        with open(os.path.join(output_dir, 'dt25.pkl'), 'rb') as f:
            pred_result = pickle.load(f)
            pred_results[f'v{infer_version}'] = pred_result
    
    # Fusing predicting results 
    pred_result_ens = fuse(pred_results, avg_gap_break_pts)
    
    # Dump outputs of fusion locally
    print("Start dumping output objects locally...")
    setup_local_dump('inference')
    with open(f"./output/pred_results/dt25.pkl", 'wb') as f:
        pickle.dump(pred_result_ens, f)
    # Dump final ranking directly to facilitate efficient submission
    pred_result_ens['y_pred_fuse'].to_csv("./output/submission.csv", 
                                          index=False)
    print("Done!!")
    
    # Push fusion results to Wandb
    print("Start pushing fusion results to Wandb...")
    output_entry = wandb.Artifact(name=f'multi_horiz_fuse', 
                                  type='output')
    output_entry.add_dir("./output/pred_results")
    exp.log_artifact(output_entry)
    print("Done!!")
    
    print("=====Finish=====")
    exp.finish()

if __name__ == '__main__':
    args = parseargs()
    main(args)