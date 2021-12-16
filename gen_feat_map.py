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

import pandas as pd 
import numpy as np

from metadata import *

# Variable definitions 
FEAT_MAP_BASE = [col for col in COLS if col not in CAT_FEATURES]


def get_feat_map(chid, gp):
    '''Return feature map for a single client.
    
    Parameters:
        chid: int, client identifier
        gp: pd.DataFrame, feature map raw data of the given client
        
    Return:
        chid: int, client identifier
        feat_map: ndarray, feature map with axes shop_tag and dt
    '''
    feat_map = (gp.pivot(index='dt', columns='shop_tag', 
                                 values=feat)
                  .reindex(index=DTS, columns=SHOP_TAGS_, 
                           fill_value=0)
                  .fillna(0))
    feat_map = np.array(feat_map)
    return chid, feat_map

if __name__ == '__main__':
    for feat in FEAT_MAP_BASE:
        feat_maps_dict = {}
        df = pd.read_parquet("./data/raw/raw_data.parquet", 
                             columns=PK+[feat])
        dump_path = f"./data/processed/feat_map/{feat}.npz"
        
        chid_gps = df.groupby(by=['chid'])
        feat_maps = Parallel(n_jobs=-1)(
            delayed(get_feat_map)(chid, gp) 
            for chid, gp in tqdm(chid_gps)
        )
        
        for chid, feat_map in feat_maps:
            feat_maps_dict[chid] = feat_map  
        feat_maps_sorted = dict(sorted(feat_maps_dict.items(), 
                                       key=lambda item: item[0]))
        feat_maps_sorted = np.array([v for v in feat_maps_sorted.values()])
        np.savez_compressed(dump_path, feat_maps_sorted)
            
        del df, feat_maps_dict, chid_gps, \
            feat_maps, feat_maps_sorted