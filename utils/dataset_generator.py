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
        train_leg: bool, if the training set contains only samples with
                   legitimate shop_tags, default=False
        production: bool, if the generated dataset is used for final
                    production (i.e., online submission), default=False
            *Note: If True, no y will be generated.
        have_y: bool, if y data is given, default=True
            *Note: It's always False for final production where no y is
                   provided
        mcls: bool, if task is modelled as multi-class classification, 
              default=False
    '''
    def __init__(self, t_end, t_window=3, horizon=1, 
                 train_leg=False, production=False, have_y=True, mcls=False):
        self._t_end = t_end
        self._t_window = t_window
        self._horizon = horizon
        self._train_leg = train_leg
        self._production = production
        self._have_y = have_y
        self._mcls = mcls
        self._setup()
    
    def run(self, feats_to_use):
        '''Start running dataset generation process.
        '''
        # Generate X feature base 
        # DataFrame with (chid, shop_tag) pairs will be generated if 
        # there's no raw numeric feature given
        self._dataset = self._get_raw_n(feats_to_use['raw_n'])
#         self.pk = self._dataset.index   # Primary key for predicting report 
        
        if feats_to_use['use_cli_attrs']:
            X_cli_attrs = self._get_cli_attrs()
            self._dataset = self._dataset.join(X_cli_attrs, 
                                               on='chid', 
                                               how='left')
            del X_cli_attrs
        
        if feats_to_use['use_tifu_pred_vecs']:
            tifu_vecs = self._get_tifu_vecs(feats_to_use['tifu'])
            self._dataset = self._dataset.join(tifu_vecs, 
                                               on='chid', 
                                               how='left')
            del tifu_vecs
        
        # Add groundtruths correponding to X samples into dataset
        if self._have_y:
            self._add_gts()
            
        self.pk = self._dataset.index   # Primary key for predicting report 
        self._dataset.reset_index(inplace=True)
        
        # Drop disabled categorical features (must run through this cleaning
        # regardless of whether to use categorical or not, because there's
        # a need to clean the cat features (e.g., those from pk) in ._dataset.
        self._drop_cat(feats_to_use['use_chid'],
                       feats_to_use['chid_as_cat'],
                       feats_to_use['use_shop_tag'])
        
        if feats_to_use['use_cat']:
            # Preprocess categorical features to alleviate memory consumption of 
            # the training process using gbdt models (e.g., lgbm)
            self._proc_cat()
        else:
            self.cat_features_ = []
        
        # Record all feature names
        self.features_ = [col for col in self._dataset if col != 'make_txn']
        
    def get_X_y(self):
        X_cols = self.features_
        X = self._dataset[X_cols]
        y = None if not self._have_y else self._dataset['make_txn'] 
        
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
        *Note: If multi-class classification is specified, then only 
               return the DataFrame with unique `chid`s as indices.
        
        Parameters:
            feats: list, features to use
        
        Return:
            X_raw_n: pd.DataFrame, raw numeric features
        '''
        if self._mcls:
            # If the task is modelled as multi-class classification
            if self._production:
                # All chids must exist in production scheme, and if y exists,
                # #chids shrinks to align with chids in y.
                X_raw_n = pd.DataFrame()
                X_raw_n['chid'] = CHIDS
            else:
                # #chids reduces due to X set. To use all clients' txns in
                # predicting month, please use production scheme.
                X_raw_n = pd.read_parquet("./data/raw/raw_data.parquet",
                                          columns=['chid', 'dt'])
                X_raw_n = X_raw_n[X_raw_n['dt'] == self._t_end]
                X_raw_n.drop('dt', axis=1, inplace=True)
                X_raw_n.drop_duplicates(inplace=True, ignore_index=True)
            X_raw_n.set_index(keys=['chid'], drop=True, inplace=True)
        else:
            feats = PK + feats
            X_raw_n = fe.get_raw_n(feats, self._t_range, self._train_leg,
                                   self._production)
        
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
        X_cli_attrs = fe.get_cli_attrs(feats, self._t_end, self._production)
        
        return X_cli_attrs
    
    def _get_tifu_vecs(self, params):
        '''Return client or predicting vectors based on the concept of 
        TIFU-KNN. For more detailed information, please refer to:
        Modeling Personalized Item Frequency Information for 
        Next-basket Recommendation.
        
        To boost fe efficiency, pre-computed tifu vectors can be dumped 
        and loaded here. Then, there's no re-computation overhead.
        
        Parameters:
            params: dict, hyperparemeters of TIFU-KNN
        
        Return:
            tifu_vecs: pd.DataFrame, client or predicting vector for 
                       each client containing either only legitimate 
                       shop_tags or all
        '''
        # Get client vector representation for each client
        purch_map_path = "./data/processed/purch_maps.pkl"
        cli_vecs = fe.get_cli_vecs(purch_map_path=purch_map_path,
                                   t1=params['t_lower_bound'], 
                                   t2=self._t_end, 
                                   gp_size=params['gp_size'],
                                   decay_wt_g=params['decay_wt_g'], 
                                   decay_wt_b=params['decay_wt_b'])
        
        if params['scale'] == 'cli':
            tifu_vecs = cli_vecs
        elif params['scale'] == 'pred':
            pred_vecs = fe.get_pred_vecs(cli_vecs=cli_vecs, 
                                         n_neighbor_candidates=params[
                                             'n_neighbor_candidates'
                                         ],
                                         sim_measure=params['sim_measure'],
                                         k=params['k'],
                                         alpha=params['alpha'])
            tifu_vecs = pred_vecs
        
        tifu_vecs = pd.DataFrame.from_dict(tifu_vecs, orient='index')
        if params['leg_only']:
            # Only dimensions corresponding to legitimate shop_tags in tifu 
            # vectors will be retained
            leg_shop_tag_indices = np.array(LEG_SHOP_TAGS) - 1
            tifu_vecs = tifu_vecs.iloc[:, leg_shop_tag_indices]
        tifu_vecs.columns = [f'tifu_shop_tag{i+1}' for i in tifu_vecs.columns]
    
        return tifu_vecs
        
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
        if self._mcls:
            y.set_index(keys=['chid'], drop=True, inplace=True)
            if self._train_leg:
                # shop_tags are presented in y in multi-class case, that's why
                # shop_tag specification isn't implemented in _get_raw_n().
                y = y[y['shop_tag'].isin(LEG_SHOP_TAGS)]
                y['shop_tag'] = y['shop_tag'].replace(LEG_SHOP_TAG_MAP)
            else:
                y['shop_tag'] = y['shop_tag'] - 1
            y.columns = ['make_txn']   # y values are indices of shop_tags
                                       # (i.e., orig_shop_tag - 1)
            self._dataset = self._dataset.join(y, how='right')
            self._dataset.dropna(inplace=True)
        else:
            y.set_index(keys=['chid', 'shop_tag'], drop=True, inplace=True)
            y['make_txn'] = 1   # Assign 1 for transactions made in the month 
                                # we want to predict

            # Add groundtruths by joining with the transaction records of the 
            # month we want to predict
            self._dataset = self._dataset.join(y, how='left')
            self._dataset.fillna(0, inplace=True)   # Assign 0 for shop_tags 
                                                    # not bought by each cli. 
        self._dataset['make_txn'] = self._dataset['make_txn'].astype(np.int8)
        
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
            try:
                self._dataset.drop('shop_tag', axis=1, inplace=True)
            except:
                # For the case that `shop_tag` isn't used as features, (e.g.,
                # multi-class classification),
                print("shop_tag doesn't exist in dataset!")
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
            elif CAT_FEAT_LBOUNDS[cat_feat] == 0:
                pass
            else:
                self._dataset[cat_feat] = (self._dataset[cat_feat] - 
                                           CAT_FEAT_LBOUNDS[cat_feat])
                
            # Convert dtypes to shrink down memory consumption and 
            # also follow the advanced topics introduced in lgbm 
            # document accessible in doc string above
            if cat_feat == 'chid':
                self._dataset['chid'] = (self._dataset['chid']
                                             .astype(np.int32))
            else:
                self._dataset[cat_feat] = (self._dataset[cat_feat]
                                                 .astype(np.int8))
