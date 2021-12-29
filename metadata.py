'''
TBrain Esun AI metadata.
Author: JiaWei Jiang

This file contains the definitions of metadata used globally throughout
the whole project.
'''
# Import packages 
import numpy as np
from itertools import product

# Metadata definitions 
N_LINES = 32975654   # #Samples 
N_CLIENTS = int(5e5)   # #Clients 
N_MONTHS = 24   # #Months 
N_SHOP_TAGS = 49   # #Shop tags
COLS = ['dt', 'chid', 'shop_tag', 'txn_cnt', 'txn_amt', 'domestic_offline_cnt',
        'domestic_online_cnt', 'overseas_offline_cnt', 'overseas_online_cnt',
        'domestic_offline_amt_pct', 'domestic_online_amt_pct',
        'overseas_offline_amt_pct', 'overseas_online_amt_pct', 'card_1_txn_cnt',
        'card_2_txn_cnt', 'card_3_txn_cnt', 'card_4_txn_cnt', 'card_5_txn_cnt',
        'card_6_txn_cnt', 'card_7_txn_cnt', 'card_8_txn_cnt', 'card_9_txn_cnt',
        'card_10_txn_cnt', 'card_11_txn_cnt', 'card_12_txn_cnt',
        'card_13_txn_cnt', 'card_14_txn_cnt', 'card_other_txn_cnt',
        'card_1_txn_amt_pct', 'card_2_txn_amt_pct', 'card_3_txn_amt_pct',
        'card_4_txn_amt_pct', 'card_5_txn_amt_pct', 'card_6_txn_amt_pct',
        'card_7_txn_amt_pct', 'card_8_txn_amt_pct', 'card_9_txn_amt_pct',
        'card_10_txn_amt_pct', 'card_11_txn_amt_pct', 'card_12_txn_amt_pct',
        'card_13_txn_amt_pct', 'card_14_txn_amt_pct', 'card_other_txn_amt_pct',
        'masts', 'educd', 'trdtp', 'naty', 'poscd', 'cuorg', 'slam',
        'gender_code', 'age', 'primary_card']   # All column names
PK = ['dt', 'chid', 'shop_tag']   # Primary key
CAT_FEATURES = ['dt', 'chid', 'shop_tag', 'masts', 
                'educd', 'trdtp', 'naty', 'poscd', 
                'cuorg', 'gender_code', 'age', 'primary_card']   # Categorical features
CAT_FEAT_LBOUNDS = {
    'dt': 1, 'chid': 1e7, 'shop_tag': 1, 'masts': 0, 
    'educd': 0, 'trdtp': 0, 'naty': 0, 'poscd': 0, 
    'cuorg': 0, 'gender_code': -1, 'age': 0, 'primary_card': 0
}   # Lower bounds of categorical features, facilitating the adjustment of cat index
CLI_ATTRS = ['chid', 'masts', 'educd', 'trdtp', 
             'naty', 'poscd', 'cuorg', 'gender_code', 
             'age']   # Client attributes
FILE_ORDERED = [f'd{int(iteration*10e4)}.parquet' for iteration in range(1, 330)] + \
               ['d32975653.parquet']   # Ordered file names of partitioned raw data
DTS = [_ for _ in range(1, 25)]   # Values of time indicators
DTS_BASE = np.array([DTS]).T   # Time indicator vector for feature engineering 
SHOP_TAGS = [_ for _ in range(49, 0, -1)]   # Values of shop tags (reverse)
SHOP_TAGS_ = [i for i in range(1, 50)]   # Values of shop tags 

LEG_SHOP_TAGS = [2, 6, 10, 12, 13, 15, 18, 19, 
                 21, 22, 25, 26, 36, 37, 39, 48]   # Legitimate 'shop_tag' in submission
LEG_SHOP_TAGS_INDICES = np.array(LEG_SHOP_TAGS) - 1
LEG_SHOP_TAG_MAP = {
    v: k for k, v in enumerate(LEG_SHOP_TAGS)
}   # Mapping from legitimate shop_tags to indices starting from zero to n-1,
    # where n is the total number of shop_tags
CHIDS = [int(1e7+i) for i in range(0, int(5e5))]
FINAL_PRODUCTION_PKS = list(product(CHIDS, LEG_SHOP_TAGS))   # All (chid, leg_shop_tag)
                                                             # pairs for final production

# Engineered features
PCT2AMTS = ['domestic_offline_txn_amt', 'domestic_online_txn_amt', 
            'overseas_offline_txn_amt', 'overseas_online_txn_amt', 
            'card_1_txn_txn_amt', 'card_2_txn_txn_amt', 'card_3_txn_txn_amt', 
            'card_4_txn_txn_amt', 'card_5_txn_txn_amt', 'card_6_txn_txn_amt', 
            'card_7_txn_txn_amt', 'card_8_txn_txn_amt', 'card_9_txn_txn_amt', 
            'card_10_txn_txn_amt', 'card_11_txn_txn_amt', 'card_12_txn_txn_amt',
            'card_13_txn_txn_amt', 'card_14_txn_txn_amt', 'card_other_txn_txn_amt']
FEAT_PRED_PARAM_KEYS = ['scale', 't_window', 'gp_size', 'decay_wt_g', 
                        'decay_wt_b', 'alpha', 'sim_deter', 'sim_measure', 
                        'k', 'n_neighbor_candidates', 'leg_only', 'shop_tag_slctn']
AMT_CNT_INTER = ['txn_apc', 'txn_apc_state', 'domestic_offline_txn_apc',
                 'domestic_offline_txn_apc_state', 'domestic_online_txn_apc',
                 'domestic_online_txn_apc_state', 'overseas_offline_txn_apc',
                 'overseas_offline_txn_apc_state', 'overseas_online_txn_apc',
                 'overseas_online_txn_apc_state', 'c1_txn_apc', 'c1_txn_apc_state',
                 'c2_txn_apc', 'c2_txn_apc_state', 'c3_txn_apc', 'c3_txn_apc_state',
                 'c4_txn_apc', 'c4_txn_apc_state', 'c5_txn_apc', 'c5_txn_apc_state',
                 'c6_txn_apc', 'c6_txn_apc_state', 'c7_txn_apc', 'c7_txn_apc_state',
                 'c8_txn_apc', 'c8_txn_apc_state', 'c9_txn_apc', 'c9_txn_apc_state',
                 'c10_txn_apc', 'c10_txn_apc_state', 'c11_txn_apc',
                 'c11_txn_apc_state', 'c12_txn_apc', 'c12_txn_apc_state',
                 'c13_txn_apc', 'c13_txn_apc_state', 'c14_txn_apc',
                 'c14_txn_apc_state', 'cother_txn_apc', 'cother_txn_apc_state']
CLI_ATTR_ABBRS = {
    'masts': 'm', 'educd': 'e', 'trdtp': 't', 'naty': 'n', 
    'poscd': 'p', 'cuorg': 'c', 'gender_code': 'g', 'age': 'a'
}
APC_STATE_ABBRS = {
    'txn_apc_state': 'x', 'domestic_offline_txn_apc_state': 'd1', 'domestic_online_txn_apc_state': 'd2',
    'overseas_offline_txn_apc_state': 'o1', 'overseas_online_txn_apc_state': 'o2', 'c1_txn_apc_state': 'c1',
    'c2_txn_apc_state': 'c2', 'c3_txn_apc_state': 'c3', 'c4_txn_apc_state': 'c4',
    'c5_txn_apc_state': 'c5', 'c6_txn_apc_state': 'c6', 'c7_txn_apc_state': 'c7',
    'c8_txn_apc_state': 'c8', 'c9_txn_apc_state': 'c9', 'c10_txn_apc_state': 'c10',
    'c11_txn_apc_state': 'c11', 'c12_txn_apc_state': 'c12', 'c13_txn_apc_state': 'c13',
    'c14_txn_apc_state': 'c14', 'cother_txn_apc_state': 'cot'
}
