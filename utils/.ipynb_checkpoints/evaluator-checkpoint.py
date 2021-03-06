'''
TBrain Esun AI evaluator.
Author: JiaWei Jiang

This file includes the definition of evaluator used to evaluate how 
well model performs.
'''
# Include packages
import os 
from tqdm import tqdm
import math
import pickle

import pandas as pd 
import numpy as np

from metadata import *

class Evaluator:
    NDCG_base = np.array(
                    [math.log(r+1, 2) for r in range(1, 4)]
                )   # Denominator of DCG and iDCG
    
    def __init__(self, data_path, pred, t_next):
        self.data_path = data_path
        self.pred = pred
        self.t_next = t_next   # Time slot to evaluate
        self._setup()
        
    def evaluate(self):
        '''Run the evaluation and return the performance.
        *****Solve bottleneck
        '''
        n_clients = 0   # #Legitimate clients to evaluate on 
                        # Suppose that clients not purchasing anything 
                        # in the time slot are excluded.
        NDCG = 0
        for chid, txn_amt in tqdm(self.gt_map.items()):
            if (txn_amt == np.zeros(3)).sum() == 3:
                # The client doesnt't purchase at t_next
                continue
            else:
                n_clients += 1
                txn_amt_pred = np.zeros(3)
                shop_tags_pred = self.pred[chid]
                txn_amt_all = self.txn_amt_true[chid]
                for i, (rank, shop_tag) in enumerate(shop_tags_pred.items()):           
                    txn_amt_pred[i] = txn_amt_all[shop_tag]
                    
                NDCG += Evaluator._NDCG_c(txn_amt, txn_amt_pred)
            NDCG_avg = NDCG / n_clients
        
        return NDCG_avg
    
    def _setup(self):
        '''
        '''
        # Prepare raw data to retrieve transaction amount
        self.df = pd.read_parquet(self.data_path, 
                                  columns=['dt', 'chid', 'shop_tag', 'txn_amt'])
        try:
            self.df = self.df[self.df['dt'] == self.t_next]
        except:
            raise ValueError("Please enter time point within"
                             "time interval [1, 24]...")
        self.df.drop('dt', axis=1, inplace=True)
        self.df = self.df.pivot(index='chid', columns='shop_tag', values='txn_amt')
        self.df.fillna(value=0, inplace=True)
        self.txn_amt_true = self.df.to_dict(orient='index')
        
        # Setup groundtruth
        if self._check_gt_existed():
            with open(f"./data/gt/t_{self.t_next}.pkl", 'rb') as f:
                self.gt_map = pickle.load(f)
        else:
            self._new_gt()
        
        # Adjust prediction format
        self.pred.set_index('chid', inplace=True)
        self.pred = self.pred.to_dict(orient='index')
    
    def _check_gt_existed(self):
        '''Check if groundtruth at the time slot to evaluate has been 
        dumped already or not. 
        '''
        gt_path = os.path.join("./data/gt", f't_{self.t_next}.pkl')
        if os.path.exists(gt_path):
            return True
        else:
            return False
    
    def _new_gt(self):
        '''Setup the groundtruth corresponding to the time slot to 
        evaluate.
        '''
        self.gt = self.gt[self.gt['shop_tag'].isin(LEG_SHOP_TAGS)]
        self.gt.sort_values(by=['chid', 'txn_amt'], ascending=False, inplace=True)
        self.gt_map = {}
        for chid in tqdm(sorted(CHIDS)):
            sub_df = self.gt[self.gt['chid'] == chid]

            if len(sub_df) == 0:
                # No consumption records at t_next
                self.gt_map[chid] = np.zeros(3)
                continue
            amt_ranking = sub_df['txn_amt'].iloc[:3]
            if len(amt_ranking) < 3:
                amt_ranking = np.pad(amt_ranking,
                                     (0, 3-len(amt_ranking)),
                                     mode='constant',
                                     constant_values=0)
            self.gt_map[chid] = np.array(amt_ranking)
    
    @classmethod
    def _NDCG_c(cls, txn_amt, txn_amt_pred):
        DCG_c = np.sum(txn_amt_pred / cls.NDCG_base)
        iDCG_c = np.sum(txn_amt / cls.NDCG_base)
        NDCG_c = DCG_c / iDCG_c
        
        return NDCG_c