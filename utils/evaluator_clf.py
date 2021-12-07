'''
Classification evaluator.
Author: JiaWei Jiang

This file includes the definition of evaluator used to evaluate how 
well model performs on classfication task.
'''
# Include packages
import os 
import pickle

import numpy as np
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, fbeta_score
from sklearn.metrics import roc_auc_score

class EvaluatorCLF:
    '''Evaluator for the first level task, classification.
    
    Parameters:
        metrics: list, evaluation metrics to use
        pos_thres: float, threshold above which the observation is 
                   classified as positive 
    '''
    def __init__(self, metrics, pos_thres):
        self._metric_names = metrics
        self.pos_thres = pos_thres
        self._metrics = {}
        self.prf = {}
        self._build_metrics()
        
    def evaluate(self, y_true, y_pred):
        '''Start evaluation.
        
        Parameters:
            y_true: ndarray, groundtruths
            y_pred: ndarray, predicting results
        '''
        y_pred = np.where(y_pred > self.pos_thres, 1, 0)
        self._cal_cf_mat(y_true, y_pred)
        for metric_name, metric in self._metrics.items():
            self.prf[metric_name] = metric(y_true, y_pred)
        
        return self.prf
    
    def _build_metrics(self):
        '''Build evaluation metric objects.
        '''
        for metric in self._metric_names:
            if metric == 'acc':
                self._metrics['acc'] = accuracy_score
            elif metric == 'precision':
                self._metrics['precision'] = precision_score
            elif metric == 'recall':
                self._metrics['recall'] = recall_score
            elif metric == 'f1':
                self._metrics['f1'] = f1_score
#             elif metric == 'fbeta':
#                 self._metrics['fbeta'] = 
            elif metric == 'auc':
                self._metrics['auc'] = roc_auc_score
                
    def _cal_cf_mat(self, y_true, y_pred):
        '''Calculate the confusion matrix and set performance stats
        facilitating the downstream matric derivation.
        
        Parameters:
            y_true: ndarray, groundtruths
            y_pred: ndarray, predicting results
        '''
        self._cf_mat = confusion_matrix(y_true, y_pred)
        
        # Retrieve performance stats
        self.tn, self.fp, self.fn, self.tp = self._cf_mat.ravel() 
        
    def _FPR(self):
        '''False Positive Rate (Type I error), rate of predicting 
        something when it isn't.
        '''
        fpr = self.fp / (self.fp+self.tn)
        
        return fpr
        
    def _FNR(self):
        '''False Negative Rate (Type II error), rate of not predicting 
        something when it is.
        
        Example:
            Fraction of missed fraudulent transactions.
        '''
        fnr = self.fn / (self.fn+self.tp)
        
        return fnr 
        
    def _TNR(self):
        '''True Negative Rate (Specificity), rate of negative samples
        classified as negative.
        '''
        tnr = self.tn / (self.tn+self.fp)
        
        return tnr
    
    def _NPV(self):
        '''Negative Predictive Value, how many predictions out of all
        negative predictions are correct.
        '''
        npv = self.tn / (self.tn+self.fn)
        
        return npv
    
    def _FDR(self):
        '''False Discovery Rate, how many predictions out of all
        positive predictions are incorrect.
        
        c.f. Precision:
            FDR = 1 - Precision
        '''
        fdr = self.fp / (self.fp+self.tp)
        
        return fdr
    
    def _TPR(self):
        '''True Positive Rate (Recall, Sensitivity), rate of positive 
        samples classified as positive.
        '''
        tpr = self.tp / (self.tp+self.fn)
        
        return tpr
    
    def _PPV(self):
        '''Postive Predictive Value (Precision), how many predictions 
        out of all positive predictions are correct. 
        '''
        ppv = self.tp / (self.tp+self.fp)
        
        return ppv
    
    def _ACC(self):
        '''Accuracy, how many predictions are correctly classified.
        '''
        acc = (self.tp+self.tn) / (self.tp+self.tn+self.fp+self.fn)
        
        return acc