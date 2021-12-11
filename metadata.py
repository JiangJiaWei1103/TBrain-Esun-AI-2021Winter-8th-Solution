'''
TBrain Esun AI metadata.
Author: JiaWei Jiang

This file contains the definitions of metadata used globally throughout
the whole project.
'''
# Import packages 
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
SHOP_TAGS = [_ for _ in range(49, 0, -1)]   # Values of shop tags
LEG_SHOP_TAGS = [2, 6, 10, 12, 13, 15, 18, 19, 
                 21, 22, 25, 26, 36, 37, 39, 48]   # Legitimate 'shop_tag' in submission
LEG_SHOP_TAG_MAP = {
    v: k for k, v in enumerate(LEG_SHOP_TAGS)
}   # Mapping from legitimate shop_tags to indices starting from zero to n-1,
    # where n is the total number of shop_tags
CHIDS = [int(1e7+i) for i in range(0, int(5e5))]
FINAL_PRODUCTION_PKS = list(product(CHIDS, LEG_SHOP_TAGS))   # All (chid, leg_shop_tag)
                                                             # pairs for final production
                                                            