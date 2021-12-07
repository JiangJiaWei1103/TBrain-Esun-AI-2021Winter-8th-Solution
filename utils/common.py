'''
Commonly used utility functions.
Author: JiaWei Jiang

This file includes some commonly used utility functions in modeling
process.
'''
# Import packages 
import os 

import yaml

# Utility function definitions
def load_cfg(cfg_path):
    '''Load and return the specified configuration.
    
    Parameters:
        cfg_path: str, path of the configuration file
    
    Return:
        cfg: dict, configuration 
    '''
    with open(cfg_path, 'r') as f:
        cfg = yaml.full_load(f)
    return cfg