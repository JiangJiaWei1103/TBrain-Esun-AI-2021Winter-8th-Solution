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

import pandas as pd 
import numpy as np 

from metadata import *

class DataSampler:
    '''Return data sample weight vector considering the information of
    historical transaction behavior revealed by `txn_amt`.
    
    Parameters:
        t_end: int, the last time point taken into consideration when 
               generating X data
        mode: str, scheme used to assign sample weight, default='naive'
        imputation: str, way to impute missing values, default='median'
            *Note: Inputation takes place when there's no historical 
                   information can be used to create sample weights
                   (e.g., shop_tag in y is the 'first' txn of client)
        scale_method: str, scaling method used to eliminate dominance 
                      effect of those always make transactions with 
                      higher `txn_amt`, default='max'
    '''
    def __init__(self, t_end, mode='naive', imputation='median',
                 scale_method='max'):
        self._t_end = t_end
        self._mode = mode
        self._imputation = imputation
        self._scale_method = scale_method
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
        
        for chid, shop_tag in tqdm(zip(chids, shop_tags_)):
            sample_weight.append(self._weights[chid][shop_tag])
        sample_weight = self._scale(chids, sample_weight)
            
        return sample_weight
    
    def _setup(self):
        '''Prepare data used to generate sample weights.
        '''
        self._df = pd.read_parquet("./data/raw/raw_data.parquet",
                                   columns=PK+['txn_cnt', 'txn_amt'])
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