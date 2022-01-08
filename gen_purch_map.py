'''
Purchasing map generator.
Author: JiaWei Jiang

This script file is used to generate purchasing map for each client.
'''
# Import packages
import os
import pickle
from tqdm import tqdm

import pandas as pd 
import numpy as np 

from paths import *
from metadata import *

# Variable definitions
DT_INDICES = [t+1 for t in range(N_MONTHS)]
EMPTY_BASKET = np.zeros(N_SHOP_TAGS, dtype=np.int8)

# Utility function definitions
def get_purch_vec(chid_dt_gp):
    '''Return purchasing vector of one client for one month.
    
    Parameter:
        chid_dt_gp: pd.DataFrame, purchasing record of one client for 
                    one month
    
    Return:
        purch_vec: ndarray, purchasing vector indicating purchasing the 
                   shop_tag or not (i.e., represented by 0/1)
    '''
    purch_vec = np.zeros(49, dtype=np.int8)
    purch_vec[chid_dt_gp['shop_tag'].values-1] = 1
    
    return purch_vec
    
def get_purch_map(chid_gp):
    '''Return purchasing map of one client (including all months).
    
    Parameter:
        chid_dt_gp: pd.DataFrame, purchasing record of one client
    
    Return:
        purch_map: ndarray, purchasing map indicating purchasing the 
                   shop_tag or not (i.e., represented by 0/1)
    '''
    purch_map = chid_gp.groupby(by=['dt']).apply(get_purch_vec)
    empty_dts = [dt-1 for dt in DT_INDICES 
                 if dt not in purch_map.index]   # -1 to align with insert idx
    purch_map = purch_map.values   # Shape (24, )
    purch_map = np.vstack(purch_map)
    # Insert empty basket index by index, becasuse inserting all at once leads 
    # to undesired results, for more information, see np.insert
    for empty_dt in empty_dts:
        purch_map = np.insert(purch_map, 
                              obj=empty_dt, 
                              values=EMPTY_BASKET, 
                              axis=0)
    
    return purch_map

if __name__ == '__main__':
    df = pd.read_parquet(DATA_PATH_RAW, columns=PK)
    chid_gps = df.groupby(by=['chid'])
    purch_maps = {}
    for chid_gp_name, chid_gp in tqdm(chid_gps):
        purch_maps[chid_gp_name] = get_purch_map(chid_gp)

    with open("./data/processed/purch_maps.pkl", 'wb') as f:
        pickle.dump(purch_maps, f)