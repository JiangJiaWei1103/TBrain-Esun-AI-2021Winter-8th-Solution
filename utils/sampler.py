'''
Data sampler (weighter).
Author: JiaWei Jiang

This file defines the data sampler to assign each training sample with
(chid, shop_tag)-specific sample weight, passing on information of
txn_amt from the perspective of 'features' to 'sampling' in training 
process.
'''
# Import packages
import os 
from tqdm import tqdm
import math

import pandas as pd 
import numpy as np 
from sklearn.utils.class_weight import compute_class_weight

from metadata import *

class DataSampler:
    '''Return data sample weight vector considering the information of
    historical transaction behavior revealed by `txn_amt`.
    
    Parameters:
        t_end: int, the last time point taken into consideration when 
               generating X data
        horizon: int, predicting step
        ds_cfg: dict, configuration for data sampler, commonly used 
                parameters are introduced as follows:
            mode-> str, scheme used to assign sample weight
            imputation-> str, way to impute missing values
                *Note: Impute when there's no historical information 
                       can be used to create sample weights (e.g., 
                       shop_tag in y is the 'first' txn of client)
            scale_method-> str, method used to eliminate dominance 
                           effect of those always make txns with 
                           higher `txn_amt`
            add_glob_cls_wt-> bool, whether to take class (shop_tag)
                              weight into consideration
        production: bool, whether to train model with all clients
                    having txns in y
            *Note: If True, then chids will align with y's chids
    '''
    def __init__(self, t_end, horizon, ds_cfg, production=True):
        self._t_end = t_end
        self._horizon = horizon
        self._mode = ds_cfg['mode']
        self._imputation = ds_cfg['imputation']
        self._scale_method = ds_cfg['scale_method']
        self._add_glob_cls_wt = ds_cfg['add_glob_cls_wt']
        self._production = production
        self._setup()
        
    def run(self):
        '''Start running sample weight generation process.
        '''
        if self._mode == 'naive':
            txn_amt_mean = (self._df
                                .groupby(by=['chid', 'shop_tag'])['txn_amt']
                                .mean())
            txn_amt_mean = txn_amt_mean.reset_index(drop=False)
            weights = txn_amt_mean.pivot(index='chid', 
                                         columns='shop_tag',
                                         values='txn_amt')
            if self._production:
                # Align with chids in y
                weights = weights.join(self._chids, how='right')
        else:
            pass
        
        self._weights = self._impute(weights)
        self._weights = self._weights.to_dict(orient='index')            
    
    def get_weight(self, chids, shop_tags):
        '''Return sample weight of each (chid, shop_tag) pair in the
        given X dataset.
        
        Parameters:
            chids: pd.DataFrame.index, chid of each sample 
            shop_tags: pd.Series, shop_tag of each sample 

        Return:
            sample_weight: ndarray, sample weights
        '''
        sample_weight = []
        shop_tags_ = shop_tags.copy()
        shop_tags_.replace({shop_tag_idx: shop_tag for shop_tag_idx, shop_tag
                            in enumerate(LEG_SHOP_TAGS)}, inplace=True)
        
        # Compute class weights
        if self._add_glob_cls_wt:
            # Perform worse so far 
            global_weights = compute_class_weight('balanced', 
                                                  sorted(shop_tags_.unique()),
                                                  shop_tags_)
#         global_weights = {shop_tag: weight for shop_tag, weight in 
#                           zip(LEG_SHOP_TAGS, global_weights)}
#         global_weights = (1e5 / shop_tags_.value_counts(sort=False)).to_dict()
        
        # Record client-specific weights
        for chid, shop_tag in tqdm(zip(chids, shop_tags_)):
            sample_weight.append(self._weights[chid][shop_tag])
        sample_weight = self._scale(chids, sample_weight)
        
        # Post process weights
        sample_weight_ = []
        for shop_tag, weight in zip(shop_tags_, sample_weight):
#             sample_weight_.append(global_weights[shop_tag] * weight)
#             sample_weight_.append(global_weights[shop_tag] * (weight+1))
#             sample_weight_.append(weight * pow(1.005, global_weights[shop_tag]))
            sample_weight_.append(math.exp(weight))#pow(, weight))
    
        return np.array(sample_weight_) 
    
    def _setup(self):
        '''Prepare data used to generate sample weights.
        '''
        self._df = pd.read_parquet("./data/raw/raw_data.parquet",
                                   columns=PK+['txn_cnt', 'txn_amt'])
        if self._production:
            pred_month = self._t_end + self._horizon
            self._chids = (self._df[self._df['dt'] == pred_month]['chid']
                               .unique())
            self._chids = pd.DataFrame(index=self._chids)
        self._df = self._df[self._df['dt'] <= self._t_end]
        
    def _impute(self, weights):
        '''Impute the missing values in the weight matrix.
        
        Parameters:
            weights: pd.DataFrame, weight matrix including weight entry
                     for each (chid, shop_tag) pair
        
        Return:
            weights: pd.DataFrame, imputed weight matrix
        '''
        if self._imputation == 'median':
            # Immune to the effect of outliers
            medians = weights.median()
            weights.fillna(value=medians, inplace=True)
        elif self._imputation == 'mean':
            # Give equal importance to each sample
            mean = weights.mean()
            weights.fillna(value=mean, inplace=True)
            
        return weights
            
    def _scale(self, chids, sample_weight):
        '''Scale the sample weights to avoid the dominance effect of 
        some clients who always make a transaction with relatively 
        large `txn_amt` compared with others.
        
        Scaling is implemented client by client without the effect of 
        dominance across different clients.
        
        Parameters:
            chids: pd.DataFrame.index, chid of each sample 
            sample_weight: list, sample weights of (chid, shop_tag)
        
        Return:
            sample_weight_scaled: ndarray, scaled sample weights
        '''
        sample_weight_ = pd.DataFrame.from_dict({'chid': chids,
                                                 'weight': sample_weight})
        if self._scale_method == 'max':
            scaling_factors = (sample_weight_.groupby(by=['chid'])['weight']
                                             .transform('max')
                                             .values)
        
        
        sample_weight_scaled = sample_weight / scaling_factors
        
        return sample_weight_scaled