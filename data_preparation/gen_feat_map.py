'''
Feature map generator.
Author: JiaWei Jiang

This script file is used to generate all the feature maps of all chids,
facilitating the further feature engineering process (e.g., spatial-
temporal pattern extraction).
'''
# Import packages
import os 
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
import json
import argparse

import pandas as pd 
import numpy as np

from paths import *
from metadata import *

# Variable definitions 
FEAT_MAP_BASE = [col for col in COLS if col not in CAT_FEATURES]

def parseargs():
    '''Parse and return the specified command line arguments.
    
    Parameters:
        None
        
    Return:
        args: namespace, parsed arguments
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--feat-type', type=str, 
                           help="type of feature map")
    args = argparser.parse_args()
    
    return args

def get_feat_map(chid, gp, imp):
    '''Return feature map for a single client.
    
    Parameters:
        chid: int, client identifier
        gp: pd.DataFrame, feature map raw data of the given client
        imp: float, imputation value
        
    Return:
        chid: int, client identifier
        feat_map: ndarray, feature map with axes shop_tag and dt
    '''
    feat_map = (gp.pivot(index='dt', columns='shop_tag', 
                         values=feat)
                  .reindex(index=DTS, columns=SHOP_TAGS_)
                  .fillna(imp)
                  .replace(0, imp))
    feat_map = np.array(feat_map)
    return chid, feat_map

if __name__ == '__main__':
    if not os.path.exists("./data/processed"):
        os.mkdir("./data/processed")
        
    args = parseargs()
    feat_type = args.feat_type
    if feat_type == 'amt':
        feats = ['txn_amt']+PCT2AMTS
        dump_path = "./data/processed/feat_map_txn_amt"
        imp = 797.165663
    elif feat_type == 'pct_cnt':
        feats = FEAT_MAP_BASE
        dump_path = "./data/processed/feat_map"
        imp = 0
    if not os.path.exists(dump_path):
        os.mkdir(dump_path)
    
    for feat in feats:
        feat_maps_dict = {}
        if (feat == 'txn_amt') or 'txn_amt' not in feat:
            data_path = DATA_PATH_RAW
        else: data_path = DATA_PATH_TXN_AMTS
        df = pd.read_parquet(data_path, columns=PK+[feat])
        feat_dump_path = f"{dump_path}/{feat}.npz"
        
        chid_gps = df.groupby(by=['chid'])
        feat_maps = Parallel(n_jobs=-1)(
            delayed(get_feat_map)(chid, gp, imp) 
            for chid, gp in tqdm(chid_gps)
        )
        
        for chid, feat_map in feat_maps:
            feat_maps_dict[chid] = feat_map  
        feat_maps_sorted = dict(sorted(feat_maps_dict.items(), 
                                       key=lambda item: item[0]))
        feat_maps_sorted = np.array([v for v in feat_maps_sorted.values()])
        np.savez_compressed(feat_dump_path, feat_maps_sorted)
            
        del df, feat_maps_dict, chid_gps, \
            feat_maps, feat_maps_sorted