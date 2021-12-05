'''
Dataset generator for ML-based model.
Author: JiaWei Jiang

This file is used for generating the dataset containing X (i.e, feats) 
and y (i.e., binary targets) of a single fold, including train and val. 
'''
# Import packages
import os 
import pickle

import pandas as pd 
import numpy as np 

from metadata import * 
import fe

# Variable definitions 

class DataGenerator:
    '''Generate one fold of dataset containing X and y.
    *Note: y is None if it's not given.
    
    Convention:
        1. Parameters related to time point (e.g., t_end) align with the
           'dt' column in raw DataFrame (i.e., t_end=24 corresponds to
           dt=24 in raw data).
        2. Function names with trailing 'n' mean numeric features, while 
           those with trailing 'c' mean categorical features.
    
    Parameters:
        t_end: int, the last time point taken into consideration when 
               generating X data
        t_window: int, lookback time windoow for numeric features that 
                  are't aggregated over time axis, default=3
        horizon: int, predicting step, default=1
        production: bool, if the generated dataset is used for final
                    production (i.e., online submission)
            *Note: If True, no y will be generated.
    '''
    def __init__(self, t_end, t_window=3, horizon=1, production=False):
        self._t_end = t_end
        self._t_window = t_window
        self._horizon = horizon
        self._production = production
        self._setup()
    
    def run(self, feats_to_use):
        '''Start running dataset generation process.
        '''
        # Generate X feature base 
        # DataFrame with (chid, shop_tag) pairs will be generated if 
        # there's no raw numeric feature given
        self._dataset = self._get_raw_n(feats_to_use['raw_n'])
        
        if feats_to_use['use_cli_attrs']:
            X_cli_attrs = self._get_cli_attrs()
            self._dataset = self._dataset.join(X_cli_attrs, on='chid', how='left')
            del X_cli_attrs
            
        # Add groundtruths correponding to X samples into dataset
        self._add_gts()
        
    def get_X_y(self):
        X_cols = [col for col in self._dataset if col != 'make_txn']
        X = self._dataset[X_cols]
        y = self._dataset['make_txn']
        
        return X, y
    
    def _setup(self):
        '''Setup basic configuration.
        '''
        self._t_start = self._t_end - self._t_window + 1
        self._t_range = (self._t_start, self._t_end)
        self._pred_month = self._t_end + self._horizon
        if self._production:
            pass
        
    
    def _get_raw_n(self, feats):
        '''Return raw numeric features without aggregation given the 
        lookback time window.
        
        Parameters:
            feats: list, features to use
        
        Return:
            X_raw_n: pd.DataFrame, raw numeric features
        '''
        feats = PK + feats
        X_raw_n = fe.get_raw_n(feats, self._t_range)
        
        return X_raw_n
    
    def _get_cli_attrs(self):
        '''Return client attribute vector for each client in current 
        month; that is, client attributes at dt=t_end
        
        Parameters:
            None
        
        Return:
            X_cli_attrs: pd.DataFrame, client attrs in current month
        '''
        feats = ['dt'] + CLI_ATTRS
        X_cli_attrs = fe.get_cli_attrs(feats, self._t_end)
        
        return X_cli_attrs
    
    def _add_gts(self):
        '''Add y labels corresponding to X samples into dataset.
        
        Parameters:
            None
        
        Return:
            None
        '''
        y = pd.read_parquet("./data/raw/raw_data.parquet", columns=PK)
        y = y[y['dt'] == self._pred_month]
        y.drop('dt', axis=1, inplace=True)
        y.set_index(keys=['chid', 'shop_tag'], drop=True, inplace=True)
        y['make_txn'] = 1   # Assign 1 for transactions made in the month we 
                            # want to predict
        
        # Add groundtruths by joining with the transaction records of the month
        # we want to predict
        self._dataset = self._dataset.join(y, how='left')
        self._dataset.fillna(0, inplace=True)   # Assign 0 for shop_tags not bought
                                                # by each client 
        self._dataset['make_txn'] = self._dataset['make_txn'].astype(np.int8)
        self._dataset.reset_index(inplace=True)