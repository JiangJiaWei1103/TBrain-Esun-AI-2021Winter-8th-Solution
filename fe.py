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

import pandas as pd
import numpy as np 

from metadata import *

# Variable definitions 
SHOP_TAGS_ = [i for i in range(1, 50)]   # Distinguish with the definition 
                                         # in `metadata.py`
        
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
#     print(X_raw_n.shape)
#     X_raw_n.reset_index(level='shop_tag', inplace=True)
    
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

# TIFU-KNN
class CliVecGenerator:
    '''Generate client vector representation for a single client based 
    on concept of TIFU-KNN. For more detailed information, please refer 
    to: XXXX
    
    Parameters:
        purch_map_path: str, path to pre-dumped purchasing maps
        t1: int, time lower bound 
        t2: int, time upper bound (exclusive)
        gp_size: int, size of each purchasing submap
        decay_wt_g: float, weight decay ratio across neighbor groups
        decay_wt_b: float, weight decay ratio within each group
    '''
    def __init__(self, purch_map_path, t1, t2, 
                 gp_size, decay_wt_g, decay_wt_b):
        with open(purch_map_path, 'rb') as f:
            self.purch_maps = pickle.load(f)
        self.t1 = t1
        self.t2 = t2  
        self.gp_size = gp_size
        self.decay_wt_g = decay_wt_g
        self.decay_wt_b = decay_wt_b
        self._setup()
        
    def get_client_vec(self, chid):
        '''Return the client vector represented by fusing repeated purchase
        pattern and collaborative one.

        Parameters:
            chid: int, client identifier

        Return:
            client_vec: ndarray, client vector representation
        '''
        purch_map = self.purch_maps[chid][self.t1:self.t2]
        if self.first_gp_size != 0:
            first_gp = purch_map[:self.first_gp_size]
            first_gp = first_gp * self.wt_g[0]
            first_gp = np.einsum('ij, i->j', first_gp, self.wt_b[self.first_gp_size:])
            
        normal_gps = np.reshape(purch_map[self.first_gp_size:], 
                                self.normal_gp_shape)   
        normal_gps = np.einsum('ijk, i->jk', normal_gps, self.normal_gp_wt)
        normal_gps = np.einsum('ij, i->j', normal_gps, self.wt_b)
        client_vec = normal_gps if self.first_gp_size == 0 else first_gp + normal_gps
    
        return client_vec
    
    def _setup(self):
        self.n_baskets = self.t2 - self.t1   # See one month as one basket
                                             # time interval is like [t1, t2)
        self.n_gps = math.ceil(self.n_baskets / self.gp_size)
        self.wt_g = [pow(self.decay_wt_g, p) for p in range(self.n_gps-1, -1, -1)]
        self.wt_b = [pow(self.decay_wt_b, p) for p in range(self.gp_size-1, -1, -1)]
        
        self.first_gp_size = self.n_baskets % self.gp_size
        if self.first_gp_size == 0:
            # If each group has the same size
            self.normal_gp_shape = (self.n_gps, self.gp_size, -1)
            self.normal_gp_wt = self.wt_g
        else:
            self.normal_gp_shape = (self.n_gps-1, self.gp_size, -1)   # Ignore the first gp
            self.normal_gp_wt = self.wt_g[1:]   # Ignore the first gp
            
def get_cli_vecs(purch_map_path, t1, t2, 
                 gp_size, decay_wt_g, decay_wt_b):
    '''Return client vector representation for each client.
    
    Parameters:
        purch_map_path: str, path to pre-dumped purchasing maps
        t1: int, time lower bound 
        t2: int, time upper bound (exclusive)
        gp_size: int, size of each purchasing submap
        decay_wt_g: float, weight decay ratio across neighbor groups
        decay_wt_b: float, weight decay ratio within each group
    Return:
        cli_vecs: dict, client vector representation for each client
    '''
    cli_vec_generator = CliVecGenerator(purch_map_path=purch_map_path, 
                                        t1=t1, 
                                        t2=t2, 
                                        gp_size=gp_size, 
                                        decay_wt_g=decay_wt_g, 
                                        decay_wt_b=decay_wt_b)
    cli_vecs = {}
    for chid in tqdm(cli_vec_generator.purch_maps.keys()):
        cli_vecs[chid] = cli_vec_generator.get_client_vec(chid)
    
    return cli_vecs

def get_pred_vecs(cli_vecs, n_neighbor_candidates, sim_measure, k, alpha):
    '''Return client prediction vector representation for each client
    considering both repeated (client-specific) and collaborative
    purchasing patterns.
    
    Parameters:
        cli_vecs: dict, client vector representation for each client
        n_neighbor_candidates: int, number of neighboring candidates 
                               used to do similiarity measurement
        sim_measure: str, similarity measure criterion 
        k: int, number of nearest neighbors
        alpha: float, balance between client-specific and collaborative
               patterns
    '''
    pred = {}
    cli_map = np.array([v for v in cli_vecs.values()])
    
    for chid, target_vec in tqdm(cli_vecs.items()):
        sim_map = {}
        un = np.zeros(N_SHOP_TAGS)
        neighbor_candidates = sample(range(N_CLIENTS), n_neighbor_candidates)
        neighbor_mat = cli_map[neighbor_candidates]
        
        if sim_measure == 'cos':
            dot_sim = np.matmul(neighbor_mat, target_vec)
            target_norm = np.linalg.norm(target_vec)
            neighbor_norm = np.linalg.norm(neighbor_mat, axis=1)
            sim_vec = dot_sim / (target_norm * neighbor_norm) 
        elif sim_measure == 'ed':
            vec_sub = neighbor_mat - target_vec
            sim_vec = np.linalg.norm(vec_sub, axis=1)
        
        sim_map = {chid_: sim for chid_, sim in zip(neighbor_candidates, sim_vec)}
        sim_map = dict(sorted(sim_map.items(), 
                              key=lambda item: item[1], 
                              reverse=True))
        neighbors = list(sim_map.keys())[:k]
        
        for n in neighbors:
            un += cli_vecs[n+int(1e7)]
        un = un / k
        pred[chid] = alpha*target_vec + (1-alpha)*un
        
        del sim_map, un, neighbor_candidates, neighbor_mat, neighbors
    
    return pred

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

