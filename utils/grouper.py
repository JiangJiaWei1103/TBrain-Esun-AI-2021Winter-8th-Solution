'''
Feature grouper.
Author: JiaWei Jiang

This file defines feature grouper helping calculate groupby stats based
on the specified groupby keys with samples within time and shop_tag 
ranges on features in interest.
'''
# Import packages
import os 
import gc

import pandas as pd 
import numpy as np 

from paths import *
from metadata import *

class FeatGrouper:
    '''Group samples within specified range (time and shop_tag) into 
    different groups based on the given groupby keys, then compute 
    stats on features in interest.
    '''
    def __init__(self, t_end):
        self._data_path = ""
        self._t_end = t_end
        self._df = None
    
    def groupby_and_agg(self, keys, time_slots, shop_tags, 
                        feats, stats):
        '''Run groupby and compute the specified stats.
        
        Parameters:
            keys: dict, groupby keys, including client attributes and 
                  transaction amount and count states
            time_slots: tuple or list, stats are computed over samples 
                        in these dts
                *Note: tuple representation is for cont. time interval
                       where (t_lower, t_upper) means interval 
                           [t_end-t_lower + 1, t_end-t_upper + 1]
            shop_tags: str or list, stats are computed over samples in 
                       these shop_tag subset
            feats: list, features to derive stats
            stats: list, stats to derive
            
        Return:
            df_agg: pd.DataFrame, aggregated data with unique feature 
                    column names and groupby key indices
        '''
        df_agg = None
        gp_key_suffix = self._get_gp_key_suffix(keys)
        gp_keys = []
        for key_set in keys.values():
            gp_keys += key_set
            
        for i, feat in enumerate(feats):
            # Setup
            self._set_data_path(feat)
            self._prepare_df(keys, time_slots, shop_tags, feat)
            
            # Groupby and compute stats
            df_feat_agg = pd.DataFrame()
            for stat in stats:
                stats_fn = self._get_stats_fns(stat)
                df_feat_agg[stat] = (self._df.groupby(by=gp_keys)
                                                   .agg({feat: stats_fn}))
            
            # Rename feature column names
            cols = []
            for stat in stats:
                cols.append(f'{feat}_tl{time_slots[0]}_tu{time_slots[1]}'
                            f'_{shop_tags}_{stat}_{gp_key_suffix}')
            df_feat_agg.columns = cols
            
            # Record processed stats
            if i == 0:
                df_agg = df_feat_agg
            else: df_agg = pd.concat([df_agg, df_feat_agg], axis=1)
            
            # Free memory
            del self._df, df_feat_agg, cols
            gc.collect()
            
        df_agg.reset_index(inplace=True)
        
        return gp_keys, df_agg
    
    def _get_gp_key_suffix(self, keys): 
        '''Return organized string suffix for agg column names.
        
        Parameters:
            keys: dict, groupby keys, including client attributes and 
                  transaction amount and count states
        
        Return:
            gp_key_suffix: str, suffix for aggregated stats column name
        '''
        cli_attr_suffix = ""
        apc_state_suffix = ""
        for k in keys['cli_attrs']:
            cli_attr_suffix += CLI_ATTR_ABBRS[k]
        for k in keys['apc_states']:                
            apc_state_suffix += APC_STATE_ABBRS[k]
        gp_key_suffix = f'{cli_attr_suffix}_{apc_state_suffix}'
        
        return gp_key_suffix
    
    def _get_stats_fns(self, stats):
        '''Return stats function list. 
        
        Parameters:
            stats: list, stats to derive
        
        Return:
            stats_fn: callable, function to drive stats
            stats_fns: dict, stats name to callable functions for 
                       deriving stats (Deprecated)
        '''
        if stats == 'mean':
            return np.mean
        elif stats == 'nanmean':
            return np.nanmean
        elif stats == 'median':
            return np.median
        elif stats == 'nanmedian':
            return np.nanmedian
        elif stats == 'std':
            return np.std
        elif stats == 'nanstd':
            return np.nanstd
        elif stats.startswith('quantile'):
            q = stats.split('_')[-1]
            return lambda x: np.quantile(x, q=float(q))
        elif stats.startswith('nanquantile'):
            q = stats.split('_')[-1]
            return lambda x: np.nanquantile(x, q=float(q))
        
    def _set_data_path(self, feat):
        '''Set raw data path corresponding to the currently processed
        feature.
        
        Parameters:
            feat: str, feature name
            
        Return:
            None
        '''
        if ('txn_amt' in feat) and (feat != 'txn_amt') and ('pct' not in feat):
            self._data_path = DATA_PATH_TXN_AMTS
        elif 'apc' in feat:
            self._data_path = DATA_PATH_APC
        else: self._data_path = DATA_PATH_RAW
            
    def _prepare_df(self, keys, time_slots, shop_tags,
                    feat):
        '''Prepare raw DataFrame to use.
        
        Parameters:
            keys: dict, groupby keys, including client attributes and 
                  transaction amount and count states
            time_slots: tuple or list, stats are computed over samples 
                        in these dts
            shop_tags: str or list, stats are computed over samples in 
                       these shop_tag subset
            feat: str, feature name
    
        Return:
            None
        '''
        # Setup DataFrame PK, goupby keys and target feature series
        self._df = pd.read_parquet(DATA_PATH_RAW, 
                                   columns=PK+keys['cli_attrs'])
        apc_state_keys = pd.read_parquet(DATA_PATH_APC, 
                                         columns=keys['apc_states'])
        self._df = pd.concat([self._df, apc_state_keys], axis=1)
        feat_series = pd.read_parquet(self._data_path, columns=[feat])
        self._df[feat] = feat_series        
        
        # Retrieve samples within specified ranges
        if type(time_slots) == type((0, )):
            # Continuous time interval
            time_lower, time_upper = time_slots[0], time_slots[1]
            time_slots = [dt for dt in range(self._t_end-time_lower + 1, 
                                             self._t_end-time_upper + 1)]
        if isinstance(shop_tags, str):
            if shop_tags == 'leg':
                shop_tags = LEG_SHOP_TAGS
            elif shop_tags == 'illeg':
                shop_tags = ILLEG_SHOP_TAGS
            elif shop_tags == 'all': 
                shop_tags = SHOP_TAGS_
        self._df = self._df[self._df['dt'].isin(time_slots)]
        self._df = self._df[self._df['shop_tag'].isin(shop_tags)]
        
    