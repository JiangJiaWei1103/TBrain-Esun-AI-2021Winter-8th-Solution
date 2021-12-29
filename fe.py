'''
TBrain Esun AI feature engineering.
Author: JiaWei Jiang

This file contains definitions of feature engineering functions, which
facilitates the eda and feature engineering cycle.

*Note: Special cases where there's client not included in time interval.
'''
# Import packages
import pickle
from tqdm import tqdm
import math
from random import sample
from joblib import Parallel, delayed
import warnings

import pandas as pd
import numpy as np 

from metadata import *
from utils.tifu_knn import *
from utils.feat_vec_generator import *
from utils.grouper import *

warnings.simplefilter('ignore')

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
    avg_txn_amt_df = avg_txn_amt_df[SHOP_TAGS_]   # Reorder shop_tags
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
                           .reindex(index=SHOP_TAGS_,  columns=DTS, fill_value=0))
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
                       .reindex(index=SHOP_TAGS_, columns=DTS, fill_value=False)).values * DTS
        txn_gap_vec = get_txn_gap_vec(txn_map)
        txn_gap_vecs[chid] = txn_gap_vec
        del txn_map, txn_gap_vec

    return txn_gap_vecs

# Common raw features
def get_raw_n(feats, t_range, train_leg=False, production=False):
    '''Return raw numeric features without aggregation for each given
    (chid, shop_tag) pair.
    
    Parameters:
        feats: list, features to use
        t_range: tuple, time interval of raw data used to generate the 
                 raw numeric features; that is, data to use is bounded
                 between [t_range[0], t_range[1]]
        train_leg: bool, if the training set contains only samples with
                   legitimate shop_tags, default=False
        production: bool, whether the dataset is used for the final
                    production
            *Note: If the raw features are used for final production,
                   then all (chid, leg_shop_tag) pairs are needed (i.e.
                   500000 * 16 pairs)
    
    Return:
        X_raw_n: pd.DataFrame, raw numeric features
    '''
    t_start, t_end = t_range[0], t_range[1]
    df = pd.read_parquet("./data/raw/raw_data.parquet", 
                         columns=feats)
    df = df[(df['dt'] >= t_start) & (df['dt'] <= t_end)]
    if train_leg or production:
        df = df[df['shop_tag'].isin(LEG_SHOP_TAGS)]
    
    # Retrieve the most recent data in raw DataFrame as base
    X_raw_n = df[df['dt'] == t_end]
    X_raw_n.set_index(keys=['chid', 'shop_tag'], drop=True, inplace=True)
    if production:
        X_raw_n = X_raw_n.reindex(FINAL_PRODUCTION_PKS)
    
    # Impute NaN values in feature 'slam'   
    if 'slam' in feats:
        # 'slam' has different imputation logic with other numeric features;
        # that is, though there's no transaction record in the previous month,
        # 'slam' may still exist.
        with open("./data/processed/slam.pkl", 'rb') as f:
            slam = pickle.load(f)
            slam = slam[(slam['dt'] >=t_start) & (slam['dt'] <= t_end)]
        filled_slam_at_dt = slam[slam['dt'] == t_end]['slam']
        X_raw_n['slam'] = X_raw_n['slam'].fillna(filled_slam_at_dt)
    
    # Sequentially join the lagging features
    for i, dt in enumerate(range(*t_range)):
        lag = i+1
        X = df[df['dt'] == dt]
        X.set_index(keys=['chid', 'shop_tag'], drop=True, inplace=True)
        X.columns = [f'{col}_lag{lag}' for col in X.columns]
        if 'slam' in feats:
            filled_slam_at_dt = slam[slam['dt'] == dt]['slam']
            X[f'slam_lag{lag}'] = X[f'slam_lag{lag}'].fillna(filled_slam_at_dt)
            del filled_slam_at_dt
        X_raw_n = X_raw_n.join(X, how='left')
        del lag, X

    X_raw_n.fillna(0, inplace=True)   # Common imputation logic (0 filling)
    X_raw_n.drop([col for col in X_raw_n.columns if col.startswith('dt')],
                 axis=1, 
                 inplace=True)
#     X_raw_n.reset_index(level='shop_tag', inplace=True)
    
    return X_raw_n

def get_raw_n_mcls(feats, t_end):
    '''Return raw numeric features without aggregation for each client.
    
    Parameters:
        feats: list, features to use
        t_end: int, the last time point taken into consideration when 
               generating X data
    
    Return:
        X_raw_n: pd.DataFrame, raw numeric features
    '''
    X_raw_n = {chid: [] for chid in CHIDS}
    col_names = []
    
    for feat, cstrs in feats.items():
        print(f"Adding raw feature vector {feat}...")
        if 'txn_amt' in feat:
            feat_map_path = f"./data/processed/feat_map_txn_amt/{feat}.npz"
        else:    
            feat_map_path = f"./data/processed/feat_map/{feat}.npz"
        feat_maps = np.load(feat_map_path)['arr_0']
        
        # Set month and shop_tag constraits
        cstr_dt, cstr_shop_tag = cstrs[0], cstrs[1]
        if isinstance(cstr_dt, tuple):
            dt_lower, dt_upper = cstr_dt[0], cstr_dt[1]
            dts = [dt for dt in range(t_end-dt_lower, t_end-dt_upper)]
        else: dts = cstr_dt
        if cstr_shop_tag == 'leg':
            shop_tags = LEG_SHOP_TAGS_INDICES
        elif cstr_shop_tag == 'all': 
            shop_tags = np.array(SHOP_TAGS_) - 1
        else: shop_tags = cstr_shop_tag
        
        # Add in raw feature vectors
        for i, feat_map in tqdm(enumerate(feat_maps)):
            feat_vec = feat_map[dts, :]
            feat_vec = list(feat_vec[:, shop_tags].flatten())
            X_raw_n[int(1e7+i)] += feat_vec
            del feat_vec
        
        # Define column names
        for dt in dts:
            for shop_tag in shop_tags:
                if isinstance(cstr_shop_tag, str):
                    shop_tag_ = shop_tag
                else: shop_tag_ = shop_tag + 1
                col_names.append(f'{feat}_raw_t{dt}_s{shop_tag_}')
        del feat_maps
    
    X_raw_n = pd.DataFrame.from_dict(X_raw_n, orient='index')
    X_raw_n.columns = col_names
    X_raw_n.index.name = 'chid'
    X_raw_n.reset_index(drop=False, inplace=True)
    return X_raw_n

def get_cli_attrs(feats, t_end, production):
    '''Return client attribute vector for each client in current month;
    that is, client attributes at dt=t_end.
    
    Parameters:
        feats: list, features to use
        t_end: int, current month
        production: bool, whether the dataset is used for the final
                    production
            *Note: If the raw features are used for final production,
                   then all (chid, leg_shop_tag) pairs are needed (i.e.
                   500000 * 16 pairs)
        
    Return:
        X_cli_attrs: pd.DataFrame, client attributes in current month
    '''
    df = pd.read_parquet("./data/raw/raw_data.parquet",
                         columns=feats)
    df = df[df['dt'] == t_end]
        
    X_cli_attrs = df.drop('dt', axis=1)
    X_cli_attrs.drop_duplicates(inplace=True, ignore_index=True)
    if production:
        # Load the latest client attibute vectors to facilitate the 
        # imputation
        with open("./data/processed/cli_attrs_latest.pkl", 'rb') as f:
            cli_attrs_latest = pickle.load(f)
        missing_chids = set(CHIDS).difference(set(X_cli_attrs['chid']))
        X_cli_attrs_missing = []
        for chid in tqdm(list(missing_chids)):
            chid_attrs = cli_attrs_latest[chid]
            X_cli_attrs_missing.append(chid_attrs)
        X_cli_attrs_missing = pd.DataFrame(X_cli_attrs_missing, 
                                           columns=X_cli_attrs.columns)
        X_cli_attrs = X_cli_attrs.append(X_cli_attrs_missing, 
                                         ignore_index=True)
            
    X_cli_attrs.sort_values(by=['chid'], inplace=True)
    X_cli_attrs.set_index(keys='chid', drop=True, inplace=True)
    
    return X_cli_attrs

def get_filled_slam(df):
    '''Return complete slam values (including 24 months) for each 
    client.
    *Note: There are some clients with NaN slams originally, which are 
           imputed with zeros in processing stage, 'convert_type.py'.
        Ex: Client 10267183 with all zero slams, but she has txn record 
    
    Parameters:
        df: pd.DataFrame, raw data
    
    Return:
        filled_slam: pd.DataFrame, complete slam values including 24
                     months for each client
            *Note: Imputation technique is designed to make it more 
                   reasonable
    '''
    filled_slam = df.copy()
    filled_slam.drop_duplicates(inplace=True)
    filled_slam = filled_slam.pivot(index='chid', columns='dt', values='slam')
    
    # Imputation logic
    # 1. Forward fill with the previous 'existing' slam
    # 2. Fill with 0 for samples before the first transaction made by
    #    each client
    filled_slam.fillna(method='ffill', axis=1, inplace=True)
    filled_slam.fillna(0, inplace=True)
    filled_slam = filled_slam.melt(value_vars=DTS, 
                                   value_name='slam', 
                                   ignore_index=False)
#     filled_slam.set_index(keys=['dt'], drop=True, append=True, inplace=True)
    
    return filled_slam

def get_latest_cli_attrs(df):
    '''Return the latest client attribute vector for each client.
    
    Parameters:
        df: pd.DataFrame, raw data
    
    Return:
        cli_attrs_latest: dict, latest client attribute vector for each
                          client
    '''
    cli_attrs_latest = {}
    df_ = df.copy()
    df_.sort_values(by=['chid', 'dt'], inplace=True)
    df_.drop_duplicates(subset=CLI_ATTRS, 
                        keep='last', 
                        inplace=True, 
                        ignore_index=True)
    
    for chid in tqdm(CHIDS):
        df_chid = df_[df_['chid'] == chid]
        df_chid.reset_index(drop=True, inplace=True)
        chid_attrs = df_chid.iloc[df_chid['dt'].idxmax()]
        cli_attrs_latest[chid] = list(chid_attrs[CLI_ATTRS])
    
    return cli_attrs_latest

# Transaction-related features
def get_txn_related_feat(t_end, feat, leg_only):
    '''Return transaction related feature vector containing specified
    information (i.e., parameter feat) for each client.
    
    Parameters:
        t_end: int, the last time point taken into consideration when 
               generating X data
        feat: str, feature name
        leg_only: bool, whether to consider legitimate shop_tags only 
    
    Return:
        txn_feat_vecs: dict, transaction related feature vector for 
                       each client
    '''
    with open("./data/processed/purch_maps.pkl", 'rb') as f:
        purch_maps = pickle.load(f)
        
    # Switch generating functions for the specified feature
    if feat == 'gap_since_first':
        txn_fn = get_gap_since_first_txn
    elif feat == 'gap_since_last':
        txn_fn = get_gap_since_last_txn
    elif feat == 'avg_gap':
        txn_fn = get_avg_txn_gap_vec
    elif feat == 'st_tgl':
        txn_fn = get_txn_st_tgl_mat
    elif feat == 'made_ratio':
        txn_fn = get_txn_made_ratio_vec
    elif feat == 'n_shop_tags':
        txn_fn = get_n_shop_tags_vec
    elif feat == 'purch_t_end':
        txn_fn = get_purch_vec_t_end
        
    # Generate transaction feature vector for each client 
    txn_feat_vecs = {}
    if feat in ['avg_gap', 'st_tgl']:
        txn_feat_base = Parallel(n_jobs=-1)(
            delayed(txn_fn)(t_end, chid, purch_map, leg_only) 
            for chid, purch_map in tqdm(purch_maps.items())
        )
        for chid, txn_feat_vec in txn_feat_base:
            txn_feat_vecs[chid] = txn_feat_vec
        txn_feat_vecs = dict(sorted(txn_feat_vecs.items(), 
                                    key=lambda item: item[0]))
    else:
        for chid, purch_map in tqdm(purch_maps.items()):
            txn_feat_vecs[chid] = txn_fn(t_end, purch_map, leg_only)
    
    return txn_feat_vecs
        
def get_gap_since_first_txn(t_end, purch_map, leg_only):
    '''Return time gap since first transaction of each shop_tag for a 
    single client.
    
    Zeros in the vector indicate that client has made his/her first 
    txn on that shop_tag at t_end. And one hundreds indicate that 
    client hasn't made a txn on that shop_tag so far.
    
    Parameters:
        t_end: int, the last time point taken into consideration when 
               generating X data
        purch_map: ndarray, purchasing behavior matrix, recording 0/1
                   indicating if transaction is made or not 
        leg_only: bool, whether to consider legitimate shop_tags only 
    
    Return:
        gap_vec: ndarray, vector including time gap since the first 
                 transaction of each shop_tag
    '''
    # Retrieve target dimensions
    purch_map = purch_map[:t_end, :]
    purch_map = purch_map[:, LEG_SHOP_TAGS_INDICES] if leg_only else purch_map
    
    purch_map = purch_map * DTS_BASE[:t_end]
    purch_map = np.where(purch_map == 0, 25, purch_map)
    gap_vec = np.min(purch_map, axis=0)
    gap_vec = t_end - gap_vec
    gap_vec = np.where(gap_vec < 0, 100, gap_vec)    
    gao_vec = gap_vec.astype(np.int8)
    
    return gap_vec

def get_gap_since_last_txn(t_end, purch_map, leg_only):
    '''Return time gap since last transaction of each shop_tag for a 
    single client.
    
    Zeros in the vector indicate that client has made a transaction at
    t_end. And one hundreds indicate that client hasn't made a txn on
    that shop_tag so far.
    
    Parameters:
        t_end: int, the last time point taken into consideration when 
               generating X data
        purch_map: ndarray, purchasing behavior matrix, recording 0/1
                   indicating if transaction is made or not 
        leg_only: bool, whether to consider legitimate shop_tags only 
    
    Return:
        gap_vec: ndarray, vector including time gap since last txn of  
                 each shop_tag
    '''
    # Retrieve target dimensions
    purch_map = purch_map[:t_end, :]
    purch_map = purch_map[:, LEG_SHOP_TAGS_INDICES] if leg_only else purch_map   
    
    purch_map = purch_map * DTS_BASE[:t_end]
    purch_map = np.where(purch_map == 0, -100, purch_map)
    gap_vec = np.max(purch_map, axis=0)
    gap_vec = t_end - gap_vec
    gap_vec = np.where(gap_vec > 24, 100, gap_vec)    
    gao_vec = gap_vec.astype(np.int8)
    
    return gap_vec

def get_avg_txn_gap_vec(t_end, chid, purch_map, leg_only):
    '''Return vector indicating average gap between transactions of 
    each shop_tag for a single client.
    *Note: Gap indicates the reciprocal of frequency.
    
    Time gap here is a little bit different from other two txn time gap
    features, gap_since_first and gap_since_last. Gap here indicates
    #months between two consecutive transactions.
    
    ***How about only make once???
    
    For example:
        dt  20  21  22  23  24
            V   X   X   X   V   ---->  time_gap == 3
            
    Parameters:
        t_end: int, the last time point taken into consideration when 
               generating X data
        chid: int, client identifier
        purch_map: ndarray, purchasing behavior matrix, recording 0/1
                   indicating if transaction is made or not 
        leg_only: bool, whether to consider legitimate shop_tags only 
        
    Return:
        chid: int, client identifier
        gap_vec: ndarray, average transaction gap of each shop_tag
    '''
    # Retrieve target dimensions
    purch_map = purch_map[:t_end, :]
    purch_map = purch_map[:, LEG_SHOP_TAGS_INDICES] if leg_only else purch_map
    
    gap_vec = []
    purch_map = purch_map * DTS_BASE[:t_end]
    for purch_vec in purch_map.T:
        # For purchasing vector of each shop_tag
        purch_vec_ = purch_vec[purch_vec != 0]
        avg_gap = np.mean(np.diff(purch_vec_, 1) - 1)   # -1 to help interpret 
                                                        # the concept 'gap'
        gap_vec.append(avg_gap)
    gap_vec = np.array(gap_vec)
    gap_vec = np.nan_to_num(gap_vec, nan=100)

    return chid, gap_vec

def get_txn_st_tgl_mat(t_end, chid, purch_map, leg_only):
    '''Return counts of transaction state toggles, including 0/0, 0/1,
    1/0, 1/1, total 4 state transitions.
    
    Parameters:
        t_end: int, the last time point taken into consideration when 
               generating X data
        chid: int, client identifier
        purch_map: ndarray, purchasing behavior matrix, recording 0/1
                   indicating if transaction is made or not 
        leg_only: bool, whether to consider legitimate shop_tags only 
        
    Return:
        chid: int, client identifier
        st_tgl_mat: ndarray, counts of state transitions, with shape 
                    (4 * n_shop_tags, )
    '''
    # Retrieve target dimensions
    purch_map = purch_map[:t_end, :]
    purch_map = purch_map[:, LEG_SHOP_TAGS_INDICES] if leg_only else purch_map
    
    st_tgl_mat = []
    for purch_vec in purch_map.T:
        # For purchasing vector of each shop_tag
        n_10 = abs(np.sum((purch_vec - 1)[1:] * purch_vec[:-1]))
        n_01 = abs(np.sum((purch_vec - 1)[:-1] * purch_vec[1:]))
        n_11 = abs(np.sum(purch_vec[:-1] * purch_vec[1:]))
        n_00 = (t_end-1) - n_10 - n_01 - n_11
        st_tgl_mat.append([n_00, n_01, n_10, n_11])
    st_tgl_mat = np.array(st_tgl_mat).flatten()
#     st_tgl_mat = np.array(st_tgl_mat).T
    
    return chid, st_tgl_mat

def get_txn_made_ratio_vec(t_end, purch_map, leg_only):
    '''Return ratio of months txn records exist for each shop_tag for 
    a single client.
    
    Parameters:
        t_end: int, the last time point taken into consideration when 
               generating X data
        purch_map: ndarray, purchasing behavior matrix, recording 0/1
                   indicating if transaction is made or not 
        leg_only: bool, whether to consider legitimate shop_tags only 
        
    Return:
        txn_made_ratio_vec: ndarray, ratio of months txn records exist
                            for each shop_tag
    '''
    # Retrieve target dimensions
    purch_map = purch_map[:t_end, :]
    purch_map = purch_map[:, LEG_SHOP_TAGS_INDICES] if leg_only else purch_map
    
    txn_made_ratio_vec = np.sum(purch_map, axis=0) / t_end
    
    return txn_made_ratio_vec

def get_n_shop_tags_vec(t_end, purch_map, leg_only):
    '''Return number of shop_tags having txn records for each month
    for a single client. 
    
    One thing important to keep in mind is that the dimension of this 
    vector for training set and validation set will be different if 
    there's no additional processing, because t_end of val set is 
    always larger than training set.
    
    Temp solution: Hard code #months to 6.
    
    Parameters:
        t_end: int, the last time point taken into consideration when 
               generating X data
        purch_map: ndarray, purchasing behavior matrix, recording 0/1
                   indicating if transaction is made or not 
        leg_only: bool, whether to consider legitimate shop_tags only 

    Return:
        n_shop_tags_vec: ndarray, num of shop_tags having txn records 
                         for each month, with shape (6, )
    '''
    # Retrieve target dimensions
    purch_map = purch_map[:t_end, :]
    purch_map = purch_map[:, LEG_SHOP_TAGS_INDICES] if leg_only else purch_map
    
    n_shop_tags_vec = np.sum(purch_map, axis=1)
    n_shop_tags_vec = n_shop_tags_vec[-6:]
    
    return n_shop_tags_vec

def get_purch_vec_t_end(t_end, purch_map, leg_only):
    '''Directly return purchasing vector at t_end, the nearest purch
    behavior.
    
    Parameters:
        t_end: int, the last time point taken into consideration when 
               generating X data
        purch_map: ndarray, purchasing behavior matrix, recording 0/1
                   indicating if transaction is made or not 
        leg_only: bool, whether to consider legitimate shop_tags only 
        
    Return:
        purch_vec: ndarray, purchasing vector at t_end
    '''
    purch_vec = purch_map[t_end-1]   # -1 to align with index    
    purch_vec = purch_vec[LEG_SHOP_TAGS_INDICES] if leg_only else purch_vec
    
    return purch_vec