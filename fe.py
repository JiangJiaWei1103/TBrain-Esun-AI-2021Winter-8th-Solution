'''
TBrain Esun AI feature engineering.
Author: JiaWei Jiang

This file contains definitions of feature engineering functions, which
facilitates the eda and feature engineering cycle.

*Note: Special cases where there's client not included in time interval.
'''
# Import packages

import pandas as pd
import numpy as np 

from metadata import *

# Variable definitions 
DTS = [d for d in range(1, 25)]
SHOP_TAGS = [t for t in range(1, 50)]
# class FeatEngineer:
#     df = pd.DataFrame()
#     cols = []
    
#     def __init__(self):
#         pass
    
#     def _set_meta_data()
    
#     def 
        
# Utility function definitions
def get_avg_shop_tags_per_month(df, t_range):
    '''Return average #shop_tags each client consumes per month.
    
    Parameters:
        df: pd.DataFrame, raw data
        t_range: tuple, time interval used to compute the values,
                 default=(0, 23)
            *Note: t_range is actually like [t_start, t_end)
    
    Return:
        avg_shop_tags: dict, average #shop_tags for each client
    '''
    # Retrieve samples within time interval
    t_start, t_end = t_range[0], t_range[1]
    df = df[(df['dt'] >= t_start) & (df['dt'] < t_end)]
    
    avg_shop_tags = df.groupby(by=['chid', 'dt'], 
                               sort=False)['shop_tag'].nunique()
    chid = avg_shop_tags.index.droplevel('dt')
    avg_shop_tags = pd.DataFrame(avg_shop_tags.values, 
                                 index=chid,
                                 columns=['n_shop_tags'])
    avg_shop_tags = avg_shop_tags.groupby(by=avg_shop_tags.index,
                                          sort=True)['n_shop_tags'].mean()
    avg_shop_tags= avg_shop_tags.to_dict()
    
    # Handle the case if the client hasn't purchased anything so far
    chids_not_purch = set(CHIDS).difference(set(avg_shop_tags.keys()))
    for chid in chids_not_purch:
        avg_shop_tags[chid] = 0
    
    return avg_shop_tags
    
def get_avg_txn_amt_per_basket(df, t_range):
    '''Return average transaction amount of each shop_tag that each 
    client consumes per month.
    
    Parameters:
        df: pd.DataFrame, raw data
        t_range: tuple, time interval used to compute the values,
                 default=(0, 23)
            *Note: t_range is actually like [t_start, t_end)
    
    Return:
        avg_txn_amt: dict, average transaction amount of each shop_tag 
                     for each client
    '''
    # Retrieve samples within time interval
    t_start, t_end = t_range[0], t_range[1]
    df = df[(df['dt'] >= t_start) & (df['dt'] < t_end)]
    
    avg_txn_amt = {}
    avg_txn_amt_series = df.groupby(by=['chid', 'shop_tag'],
                                    sort=False)['txn_amt'].mean()
    avg_txn_amt_df = avg_txn_amt_series.unstack(level='shop_tag', fill_value=0)
    avg_txn_amt_df = avg_txn_amt_df[SHOP_TAGS]   # Reorder shop_tags
    avg_txn_amt_df.sort_index(inplace=True)
    for chid, avg_txn_amt_vec in avg_txn_amt_df.iterrows():
        avg_txn_amt[chid] = np.array(avg_txn_amt_vec)
    
    # Handle the case if the client hasn't purchased anything so far
    chids_not_purch = set(CHIDS).difference(set(avg_txn_amt.keys()))
    for chid in chids_not_purch:
        avg_txn_amt[chid] = np.zeros(N_SHOP_TAGS)

    return avg_txn_amt

def get_txn_cnt_map(df, chid=None):
    '''Return transaction count map either for client-specific
    representation or the global one (i.e., summming over clients).
    
    Parameters:
        df: pd.DataFrame, raw data
        chid: int, client identifier, default=None
            *Note: None is remained for the global representation
    
    Return:
        txn_cnt_map: ndarray, transaction count map
    '''
    # Retrieve data of the specified client
    if chid is not None:
        df = df[df['chid'] == chid]
    
    chid_gps = df.groupby(by=['chid'], sort=False)
    txn_cnt_map = np.zeros((N_SHOP_TAGS, N_MONTHS))
    for chid, gp in tqdm(chid_gps):
        txn_cnt_map_ = (gp
                           .pivot(index='shop_tag', columns='dt', values='txn_cnt')
                           .fillna(0)
                           .reindex(index=SHOP_TAGS,  columns=DTS, fill_value=0))
        txn_cnt_map += txn_cnt_map_.values
        del txn_cnt_map_
    
    return txn_cnt_map

def get_txn_gap_vecs(df):
    '''Return transaction gap vector containing the avg time gap btw
    two transactions on the specific shop_tag for each client.
    
    Parameters:
        df: pd.DataFrame, raw data
    
    Return:
        txn_gap_vecs: dict, transaction gap vector for each client
    '''
    def get_txn_gap_vec(txn_map):
        '''Return vector indicating average gap between transactions of 
        each shop_tag.
        *Note: Gap indicates the reciprocal of frequency.

        Parameters:
            txn_map: ndarray, indicating if there's a transaction on the 
                     shop_tag in the corresponding month

        Return:
            txn_gap_vec: ndarray, transaction gap of each shop_tag
        '''
        txn_gap_vec = np.array([24.0 for _ in range(N_SHOP_TAGS)])
        for i, txn_vec in enumerate(txn_map):
            txn_vec_ = txn_vec[txn_vec != 0]
            txn_gap_avg = np.mean(np.diff(txn_vec_, n=1) - 1)   # -1 to help interpret the concept 'gap'
                                                                # (e.g., continuous txn will have val 0)
            if not np.isnan(txn_gap_avg):
                txn_gap_vec[-(i+1)] = txn_gap_avg

        return txn_gap_vec
    
    chid_gps = df.groupby(by=['chid'], sort=True)
    txn_gap_vecs = {}
    for chid, gp in tqdm(chid_gps):
        txn_map = (gp
                       .pivot(index='shop_tag', columns='dt', values='txn_cnt')
                       .fillna(0)
                       .astype(bool)
                       .reindex(index=SHOP_TAGS, columns=DTS, fill_value=False)).values * DTS
        txn_gap_vec = get_txn_gap_vec(txn_map)
        txn_gap_vecs[chid] = txn_gap_vec
        del txn_map, txn_gap_vec

    return txn_gap_vecs