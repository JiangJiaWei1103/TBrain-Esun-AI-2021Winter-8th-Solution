'''
XGB booster extractor.
Author: JiaWei Jiang

This file defines the extractor saving best boosters of xgb cv method,
which is useful when users want to get oof predicting results using 
best booster in each eval fold.
'''
# Import packages

import xgboost as xgb

class XGBstExtractor(xgb.callback.TrainingCallback):
    '''Help saving best booster in each eval fold in cv process.
    
    Parameters:
        cvboosters: list, empty recorder for best cv boosters 
    '''
    def __init__(self, cvboosters):
        self._cvboosters = cvboosters
        
    def after_training(self, model):
        '''Record best cv boosters.
        '''
        for cvfold in model.cvfolds:
            self._cvboosters.append(cvfold.bst)
        
        return model