'''
This script file is used to convert dtypes of features to save memory.
Author: JiaWei Jiang
'''
# Import packages
import os 
import csv

import pandas as pd 
import numpy as np

# Variable definitions
DUMP_PATH = "./data/partitioned"
N_LINES = 32975654
COLS_WITH_NAN = ['masts', 'educd', 'trdtp', 'naty',  
                 'poscd', 'cuorg', 'slam', 'gender_code',
                 'age']
IMPUTATION = [0 for i in range(7)] + [-1, 0]
TO_REPLACE = {k: {'': dummy} for k, dummy in zip(COLS_WITH_NAN, IMPUTATION)}
DTYPES = [np.int8, np.int32, np.int8, np.int16, np.float64] + \
         [np.int16 for i in range(4)] + [np.float64 for i in range(4)] + \
         [np.int16 for i in range(15)] + [np.float64 for i in range(15)] + \
         [np.int8 for i in range(6)] + [np.float64, np.int8, np.int8, np.int8]

# Utility function definition
def dump_sub_df(sub_df, i):
    '''Simply clean sub-dataframe and dump it to local storage.
    
    Parameters:
        sub_df: list, a subset of raw dataframe 
        i: int, current iteration
    
    Return:
        None
    '''
    sub_df = pd.DataFrame(sub_df, columns=cols)
    sub_df.replace({'shop_tag': {'other': '49'}}, value=None, inplace=True)
    sub_df.replace(TO_REPLACE, value=None, inplace=True)
    sub_df = sub_df.astype(col_dtypes)

    sub_df.to_parquet(f"{DUMP_PATH}/d{i}.parquet", index=False)
    
if __name__ == '__main__':
    cols, sub_df = [], []
    col_dtyps = {}
    if not os.path.exists(DUMP_PATH):
        os.mkdir(DUMP_PATH)
    
    with open("./data/raw/tbrain_cc_training_48tags_hash_final.csv", 'r') as f:
        for i, l in enumerate(csv.reader(f)):
            if i == 0:
                cols = l
                col_dtypes = {k: v for k, v in zip(cols, DTYPES)}
                continue
            sub_df.append(l)
            if i % 10e4 == 0 or i == N_LINES-1:
                print(f"Dump sub-dataframe at iter {i}...")
                dump_sub_df(sub_df, i)
                del sub_df
                sub_df = []