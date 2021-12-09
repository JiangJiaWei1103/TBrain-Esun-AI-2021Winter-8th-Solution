'''
NBR tree-based model predicting script.
Author: JiaWei Jiang 

This file is the predicting script of tree-based ML method aiming at 
determining whether the client will make transactions on legitimate
`shop_tag`s.

Notice that this is only the first stage of the whole work (i.e.,
binary classification for downstream ranking task).
'''
# Import packages
import os
import argparse
import pickle
from time import process_time as proc_t

import yaml
import pandas as pd 
import numpy as np 
import lightgbm as lgb
import wandb

from utils.common import load_cfg
from utils.dataset_generator import DataGenerator

def parseargs():
    '''Parse and return the specified command line arguments.
    
    Parameters:
        None
        
    Return:
        args: namespace, parsed arguments
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model-name', type=str, 
                           help="model to use")
    argparser.add_argument('--model-version', type=int, 
                           help="version of the model used to predict")
    argparser.add_argument('--pred-month', type=int,
                           help="month to predict, please specify 25 to "
                                "enable the final production")
    argparser.add_argument('--pos-thres', type=float, default=0.1,
                           help="threshold above which the observation is"
                                "classified as positive")
    
    args = argparser.parse_args()
    return args

def predict(dg_cfg, model, pred_month):
    '''Run cross-validation.
    
    Parameters:
        dg_cf: dict, configuration for dataset generation
        model_name: str, model to use
        pred_month: int, month to predict
    
    Return:
        pred_result: pd.DataFrame, final predicting results
    '''
    print(f"Prediction for final production starts...")
    t_start = proc_t()
    
    print("Generating testing set...")
    t_end = pred_month - dg_cfg['horizon']
    dg_test = DataGenerator(t_end, dg_cfg['t_window'], dg_cfg['horizon'],
                            production=True, have_y=False)   
    dg_test.run(dg_cfg['feats_to_use'])
    X_test, _ = dg_test.get_X_y()
    pred_result = pd.DataFrame(index=dg_test.pk)
    
    print(f"Start inference on testing data for pred_month {pred_month}...")
    y_test_pred = model.predict(data=X_test,
                                num_iteration=model.best_iteration)
    pred_result['y_pred'] = y_test_pred
    print("Done!\n")
    
    t_elapsed = proc_t() - t_start
    print(f"Prediction for pred_month {pred_month} ends...")
    print(f"Total time consumption: {t_elapsed} sec.")

    return pred_result

def main(args):
    '''Main function for training and evaluation process.
    
    Parameters:
        args: namespace,  
    '''
    # Setup the experiment and logger
    exp = wandb.init(project='Esun',
                     name='tree-based',
                     job_type='inference')
    
    # Setup basic configuration
    model_name = args.model_name
    model_version = args.model_version
    pred_month = args.pred_month
    pos_thres = args.pos_thres
    
    dg_cfg = load_cfg("./config/data_gen.yaml")
    
    # Pull well-trained model from Wandb
    output = exp.use_artifact(f'{model_name}:v{model_version}', 
                              type='output')
    output_dir = output.download()
    with open(os.path.join(output_dir, 
                           f"models/val_month{pred_month-1}.pkl"), 'rb') as f:
        # Use model trained on the latest data among all the models
        ##if ensemble is considered, then other methods will be implemented)##
        model = pickle.load(f)
    
    # Run inference
    pred_result = predict(dg_cfg, model, pred_month)
    
    # Dump outputs of the experiment locally
    print("Start dumping output objects locally...")
    with open(f"./output/pred_results/dt{pred_month}.pkl", 'wb') as f:
        pickle.dump(pred_result, f)
    print("Done!!")
    
    # Push predicting results to Wandb
    # Note: Name of the Artifact must be unique across a project; that is, we
    #       can't use the same name for different type specification
    print("Start pushing predicting results to Wandb...")
    output_entry = wandb.Artifact(name=f'{model_name}_infer', 
                                  type='output')
    output_entry.add_dir("./output/pred_results")
    exp.log_artifact(output_entry)
    print("Done!!")
    
    print("=====Finish=====")
    exp.finish()
    
# Main function 
if __name__ == '__main__':
    args = parseargs()
    main(args)