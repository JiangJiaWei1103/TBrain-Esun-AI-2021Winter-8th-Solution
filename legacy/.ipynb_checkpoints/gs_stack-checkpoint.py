import os 
import subprocess

import numpy as np
import yaml 


if __name__ == '__main__':
    base_version_dict = {
#         'orig_top6': ([144, 47, 149, 150, 151, 152], 
#                       [19, 63, 21, 22, 23, 24]),
#         'orig_diverse': ([149, 150, 151, 152, 167, 168, 171, 172], 
#                          [21, 22, 23, 24, 30, 31, 34, 35]),
#         'illeg_all': ([v for v in range(173, 184)],
#                       [v for v in range(36, 44)]+[47, 45, 46]),
#         'illeg_top7': ([v for v in range(177, 184)],
#                        [40, 41, 42, 43, 47, 45, 46]),
#         'illeg_diverse': ([173, 175, 178, 179, 181, 182],
#                           [36, 38, 41, 42, 47, 45]),
#         'final_all': ([v for v in range(184, 196)], 
#                       [v for v in range(48, 56)]+[58, 59, 60, 61]),
#         'final_top6': ([184, 186, 187, 190, 192, 194], 
#                        [48, 50, 51, 54, 58, 60]),
#         'top_mixed': ([144, 47, 149, 150, 151, 152]+[v for v in range(177, 184)]+[184, 186, 187, 190, 192, 194],
#                       [19, 63, 21, 22, 23, 24, 40, 41, 42, 43, 47, 45, 46, 48, 50, 51, 54, 58, 60]),
#         'top_top': ([144, 47, 149, 150, 151, 152, 184, 186, 187, 190, 192, 194],
#                     [19, 63, 21, 22, 23, 24, 48, 50, 51, 54, 58, 60]),
#         'top_top2': ([144, 47, 149, 150, 151, 152, 184, 186, 190],
#                     [19, 63, 21, 22, 23, 24, 48, 50, 54]),
#         'orig_top6_with_blend': (['l144', 'l47', 'l149', 'l150', 'l151', 'l152', 'b1', 'b2'], 
#                                  ['l19', 'l63', 'l21', 'l22', 'l23', 'l24', 'b1', 'b2']),
#         'restack1': (['lm11', 'lm12', 'lm16', 'xm8', 'lm17'],
#                      ['lm7', 'lm6', 'lm10', 'xm7', 'lm11']),
#         'restack2': (['l149', 'l152', 'l167', 'l168', 'l170', 'l171', 'l173', 'l175', 'b1', 'b2'], 
#                      ['l21', 'l24', 'l30', 'l31', 'l33', 'l34', 'l36', 'l38', 'b1', 'b2']),
#         'restack3': (['lm18', 'xm8', 'xm10', 'b1', 'b2', 'b3'], 
#                      ['lm12', 'xm7', 'xm9', 'b1', 'b2', 'b3']),
#         'restack4': (['lm16', 'xm8', 'lm18', 'xm10', 'b1', 'b2', 'b3'], 
#                      ['lm10', 'xm7', 'lm12', 'xm9', 'b1', 'b2', 'b3']),
#         'restack5': (['lm18', 'xm8', 'xm10', 'b1', 'b2', 'b3'], 
#                      ['lm12', 'xm7', 'xm9', 'b1', 'b2', 'b3']),
#         'restack6': (['lm16', 'xm8', 'lm18', 'xm10', 'b1', 'b2', 'b3'], 
#                      ['lm10', 'xm7', 'lm12', 'xm9', 'b1', 'b2', 'b3']),
        'mix': (['l184', 'l186', 'l187', 'l190', 'l192', 'l194', 'l195', 'b1', 'b2', 'b3'], 
                ['l48', 'l50', 'l51', 'l54', 'l58', 'l60', 'l61', 'b1', 'b2', 'b3']),
    }
    
    restacks = [False]
    for (meta_name, base_versions), rsk in zip(base_version_dict.items(), restacks):
#         restack = 'True' if meta_name.startswith('restack') else 'False'
        oof_versions = np.array(base_versions[0]).astype(str)
        unseen_versions = np.array(base_versions[1]).astype(str)
        for meta_model_name in ['lgbm', 'xgb']:
            proc_tr = ['python', 'train_stacker.py',
                       '--meta-model-name', f'{meta_model_name}',
                       '--n-folds', '5',
                       '--eval-metrics', 'ndcg@3',
                       '--objective', 'mcls']
            if rsk:
                proc_tr.append('--restacking')
                proc_tr.append('True')
            proc_tr.append('--oof-versions')
            for v in oof_versions: proc_tr.append(v)
            subprocess.run(proc_tr)
            
            proc_pred = ['python', 'pred_stacker.py',
                         '--meta-model-name', f'{meta_model_name}',
                         '--meta-model-version', '0',   # Latest
                         '--pred-month', '25',
                         '--objective', 'mcls']
            if rsk:
                proc_pred.append('--restacking')
                proc_pred.append('True')
            proc_pred.append('--oof-versions')
            for v in oof_versions: proc_pred.append(v)
            proc_pred.append('--unseen-versions')
            for v in unseen_versions: proc_pred.append(v)
            subprocess.run(proc_pred)
