import os 
import yaml 
import subprocess

if __name__ == '__main__':
    dg_cfgs1 = [f'feat_comb/raw/data_gen{i}.yaml' for i in range(3, 9)]
    dg_cfgs2 = [f'feat_comb/txn_related/data_gen{i}.yaml' for i in range(1, 2)]
    dg_cfgs3 = [f'feat_comb/feat_vecs/data_gen{i}.yaml' for i in range(3, 5)]
    dg_cfgs_final = [f'feat_comb/final/god_set{i}.yaml' for i in range(9, 11)]
    
    for i, cfg in enumerate(dg_cfgs_final):
        with open(f"./config/{cfg}", 'r') as f:
            dg_cfg = yaml.full_load(f)
        with open("./config/data_gen.yaml", 'w') as f:
            yaml.dump(dg_cfg, f)
#         if (i != 0) and (i != 1):
        subprocess.run(['python', 'train_tree.py',
                        '--model-name', 'lgbm',
                        '--n-folds', '1',
                        '--eval-metrics', 'ndcg@3',
                        '--train-leg', 'True',
                        '--train-like-production', 'True',
                        '--val-like-production', 'True',
                        '--mcls', 'True',
                        '--eval-train-set', 'True'])
        subprocess.run(['python', 'pred_tree.py',
                        '--model-name', 'lgbm',
                        '--model-version', '0',
                        '--val-month', '24', 
                        '--pred-month', '25',
                        '--mcls', 'True'])
#             continue
        
#         if i == 0:
#             subprocess.run(['python', 'pred_tree.py',
#                             '--model-name', 'lgbm',
#                             '--model-version', '166',
#                             '--val-month', '24', 
#                             '--pred-month', '25',
#                             '--mcls', 'True'])
#         elif i == 1:
#             subprocess.run(['python', 'pred_tree.py',
#                         '--model-name', 'lgbm',
#                         '--model-version', '167',
#                         '--val-month', '24', 
#                         '--pred-month', '25',
#                         '--mcls', 'True'])
