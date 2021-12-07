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
        self.pk = self._dataset.index   # Primary key for predicting report 
        
        if feats_to_use['use_cli_attrs']:
            X_cli_attrs = self._get_cli_attrs()
            self._dataset = self._dataset.join(X_cli_attrs, on='chid', how='left')
            del X_cli_attrs
        
        if feats_to_use['use_tifu_pred_vecs']:
            with open("./data/processed/pred_vecs_dt_1-23.pkl", 'rb') as f:
                vecs = pd.DataFrame.from_dict(pickle.load(f), orient='index')
                vecs.columns = [f'tifu_shop_tag{i+1}' for i in vecs.columns]
#             vecs = self._get_tifu_vecs(feats_to_use['tifu'])
            self._dataset = self._dataset.join(vecs, on='chid', how='left')
            del vecs
        
        # Add groundtruths correponding to X samples into dataset
        self._add_gts()
        
        # Drop disabled categorical features 
        self._drop_cat(feats_to_use['use_chid'],
                       feats_to_use['chid_as_cat'],
                       feats_to_use['use_shop_tag'])
        
        # Preprocess categorical features to alleviate memory consumption of 
        # the training process using gbdt models (e.g., lgbm)
        self._proc_cat()
        
        # Record all feature names
        self.features_ = [col for col in self._dataset if col != 'make_txn']
        
    def get_X_y(self):
        X_cols = self.features_
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
        
        # Setup attibutes
        self.features_ = []
        self.cat_features_ = CAT_FEATURES.copy()   # Copy to avoid messing up 
                                                   # constant metadata access
                                                   # (notice removing of feat)
    
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
    
    def _get_tifu_vecs(self, params):
        '''Return client or predicting vectors based on the concept of 
        TIFU-KNN. For more detailed information, please refer to:
        
        Parameters:
            params: dict, hyperparemeters of TIFU-KNN
        
        Return:
            cli_vecs or pred_vecs: dict, client or predicting vector 
                                   for each client
        '''
        # Get client vector representation for each client
        cli_vecs = fe.get_cli_vecs(t1=params['t_lower_bound'], 
                                   t2=self._t_end, 
                                   gp_size=params['gp_size'],
                                   decay_wt_g=params['decay_wt_g'], 
                                   decay_wt_b=params['decay_wt_b'])
        
        if tifu['scale'] == 'cli':
            return cli_vecs
        elif tifu['scale'] == 'pred':
            pred_vecs = fe.get_pred_vecs(cli_vecs=cli_vecs, 
                                         n_neighbor_candidates=params[
                                             'n_neighbor_candidates'
                                         ],
                                         sim_measure=params['sim_measure'],
                                         k=params['k'],
                                         alpha=params['alpha'])
            return pred_vecs
    
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
        
    def _drop_cat(self, use_chid, chid_as_cat, use_shop_tag):
        '''Drop disabled categorical features.
        
        Parameters:
            use_chid: bool, whether chid is used 
            chid_as_cat: bool, whether chid is treated as categorical data
            use_shop_tag: bool, whether shop_tag is used
        '''
        self.cat_features_.remove('dt')
        self.cat_features_.remove('primary_card')
        if not use_chid:
            self._dataset.drop('chid', axis=1, inplace=True)
            self.cat_features_.remove('chid')
        elif not chid_as_cat:
            # Treat chid as a numeric feature
            self.cat_features_.remove('chid')
            
        if not use_shop_tag:
            self._dataset.drop('shop_tag', axis=1, inplace=True)
            self.cat_features_.remove('shop_tag')
            
    def _proc_cat(self): 
        '''Preprocess categorical features to alleviate the memory load
        for training process using gbdt models (e.g., lgbm).
        
        The main purpose is to make categories a list of continuous
        integers starting from 0. For more detailed information, please 
        refer to:
            https://lightgbm.readthedocs.io/en/latest/Quick-Start.html
            
        Parameters:
            None
            
        Return:
            None
        '''
        for cat_feat in self.cat_features_:
            if cat_feat == 'poscd':
                self._dataset['poscd'] = (self._dataset['poscd']
                                              .replace(99, 11))
            elif cat_feat == 'cuorg':
                self._dataset['cuorg'] = (self._dataset['cuorg']
                                              .replace([35, 38, 40], 
                                                       [10, 33, 34]))
                self._dataset['cuorg'] = (self._dataset['cuorg']
                                              .astype(np.int8))
            elif CAT_FEAT_LBOUNDS[cat_feat] == 0:
                continue
            else:
                self._dataset[cat_feat] = (self._dataset[cat_feat] - 
                                           CAT_FEAT_LBOUNDS[cat_feat])
                # Convert dtypes to shrink down memory consumption and 
                # also follow the advanced topics introduced in lgbm 
                # document accessible in doc string above
                if cat_feat == 'chid':
                    self._dataset['chid'] = (self._dataset['chid']
                                                 .astype(np.int32))
                if cat_feat == 'shop_tag':
                    self._dataset['shop_tag'] = (self._dataset['shop_tag']
                                                     .astype(np.int8))
