'''
Raw DataFrame generation.
Author: JiaWei Jiang

This script file aims at generating raw DataFrame for further analysis
and feature engineering process.
'''
# Import packages
import os 
from tqdm import tqdm

import pandas as pd 

from paths import *
from metadata import *

# Variable definition
FILE_PATH = "./data/partitioned"

# Main function
if __name__ == '__main__':
    # Generate raw DataFrame for original raw features
    df = pd.DataFrame()
    for file in tqdm(sorted(os.listdir(FILE_PATH))):
        df_ = pd.read_parquet(os.path.join(FILE_PATH, file))
        df = pd.concat([df, df_], ignore_index=True)
        del df_

    df.to_parquet(DATA_PATH_RAW, index=False)
    
    # Generate raw DataFrame for pct-columns converted to amt representation
    pct_cols = [col for col in df.columns if 'pct' in col]
    txn_amt_cols = [col.replace('amt_pct', 'txn_amt') for col in pct_cols]

    txn_amts = df[pct_cols].values * np.expand_dims(df['txn_amt'].values, 
                                                    axis=1)
    df_txn_amts = pd.DataFrame(txn_amts, columns=txn_amt_cols)
    df_txn_amts = pd.concat([df[PK], df_txn_amts], axis=1)
    df_txn_amts.to_parquet(f"./data/raw/raw_txn_amts.parquet", index=False)