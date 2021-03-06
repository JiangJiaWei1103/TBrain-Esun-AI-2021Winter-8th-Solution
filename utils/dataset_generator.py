'''
Dataset generator for ML-based model.
Author: JiaWei Jiang

This file is used for generating the dataset containing X (i.e, feats) 
and y (i.e., binary targets) of a single fold, including train and val. 
'''
# Import packages
import os 
import pickle
from tqdm import tqdm

import pandas as pd 
import numpy as np 

from metadata import * 
import utils.fe as fe

class DataGenerator:
    '''Generate one fold of dataset containing X and y.
    *Note: y is None if it's not given.
    
    Convention:
        1. Parameters related to time point (e.g., t_end) align with the
           'dt' column in raw DataFrame (i.e., t_end=24 corresponds to
           dt=24 in raw data).
        2. Function names with trailing 'n' mean numeric features, while 
           those with trailing 'c' mean categorical features.
    
    Parameters:
        t_end: int, the last time point taken into consideration when 
               generating X data
        t_window: int, lookback time windoow for numeric features that 
                  are't aggregated over time axis, default=3
        horizon: int, predicting step, default=1
        train_leg: bool, if the training set contains only samples with
                   legitimate shop_tags, default=False
        production: bool, if the generated dataset is used for final
                    production (i.e., online submission), default=False
            *Note: If True, no y will be generated.
        have_y: bool, if y data is given, default=True
            *Note: It's always False for final production where no y is
                   provided
        mcls: bool, if task is modelled as multi-class classification, 
              default=False
        drop_cold_start_cli: bool, whether to drop samples with 
                             cold-start clients present in y data 
        gen_feat_tolerance: int, tolerance of number of dts used to gen
                            each feature, default=6
        drop_zero_ndcg_cli: bool, whether to drop chids with 0 NDCGs in 
                            training set 
        rand_samples: tuple, ratio of #samples randomly sampled from the 
                      entire dataset and random state to speedup training 
                      process, default=None
    '''
    def __init__(self, t_end, t_window=3, horizon=1, 
                 train_leg=False, production=False, have_y=True, mcls=False,
                 drop_cold_start_cli=False, gen_feat_tolerance=6, 
                 drop_zero_ndcg_cli=False, rand_samples=None):
        self._t_end = t_end
        self._t_window = t_window
        self._horizon = horizon
        self._train_leg = train_leg
        self._production = production
        self._have_y = have_y
        self._mcls = mcls
        self._drop_cold_start_cli = drop_cold_start_cli
        self._gen_feat_tolerance = gen_feat_tolerance
        self._drop_zero_ndcg_cli = drop_zero_ndcg_cli
        self._rand_samples = rand_samples
        
        self._setup()
    
    def run(self, feats_to_use):
        '''Start running dataset generation process.
        '''
        # Check for configuration
        assert feats_to_use['use_cat'] == \
               feats_to_use['use_cli_attrs'], ("Categorical feats must"
               " be disabled if client attributes are disabled.")
        
        # Generate X feature base 
        # DataFrame with (chid, shop_tag) pairs will be generated if 
        # there's no raw numeric feature given
        print("Generating raw features...")
        self._dataset = self._get_raw_n(feats_to_use['raw_n'])
        
        if feats_to_use['use_cli_attrs'] or feats_to_use['use_gp_stats']:
            print("Generating client attributes...")
            X_cli_attrs = self._get_cli_attrs()
            self._dataset = self._dataset.join(X_cli_attrs, 
                                               on='chid', 
                                               how='left')
            del X_cli_attrs
        
        if feats_to_use['use_tifu_pred_vecs']:
            print(f"Generating tifu-knn {feats_to_use['tifu']['scale']} "
                  "vector...")
            for param_set in range(len(feats_to_use['tifu']['scale'])):
                params = {}
                for k, v in feats_to_use['tifu'].items():
                    params[k] = v[param_set]
                tifu_vecs = self._get_tifu_vecs(params)
                self._dataset = self._dataset.join(tifu_vecs, 
                                                   on='chid', 
                                                   how='left',
                                                   rsuffix=f'{param_set+1}')
            del params, tifu_vecs
            
        if feats_to_use['use_feat_pred_vecs']:
            feat_pred_mat = pd.DataFrame(index=self._dataset.index.values)
            feat_pred_mat.index.name = 'chid'
            
            for feat, cfgs in feats_to_use['feat'].items():
                print(f"Generating tifu-like feature vector {feat} with "
                      f"#parameter sets={len(cfgs)}...")
                for i, cfg in enumerate(cfgs):
                    # For all feature vector configurations for a single feat
                    params = {}
                    for k, v in zip(FEAT_PRED_PARAM_KEYS, cfg):
                        params[k] = v
                    feat_vecs = self._get_feat_vecs(params, feat)
                    feat_pred_mat = feat_pred_mat.join(feat_vecs, 
                                                       on='chid', 
                                                       how='left',
                                                       rsuffix=f'{i+1}')
                    del params, feat_vecs
            if feats_to_use['fp_pproc'] is not None:
                print(f"Post-processing tifu-like feature matrix...")
                feat_pred_mat = self._post_proc_fp_mat(feat_pred_mat,
                                                       feats_to_use['fp_pproc'])
            self._dataset = self._dataset.join(feat_pred_mat, 
                                               on='chid', 
                                               how='left')
            del feat_pred_mat
        
        if feats_to_use['use_txn_related_feats']:
            txn_feat_items = feats_to_use['txn_feat_candidates'].items()
            for feat, shop_tag_cstr in txn_feat_items:
                if shop_tag_cstr is None: continue
                print(f"Generating txn-related feature {feat}...")
                txn_related_feat = self._get_txn_related_feats(feat, 
                                                               shop_tag_cstr)
                self._dataset = self._dataset.join(txn_related_feat, 
                                                   on='chid', 
                                                   how='left')
                del txn_related_feat
                
        if feats_to_use['use_gp_stats']:
            fg = fe.FeatGrouper(t_end=self._t_end)
            for gp_stats_cfg in feats_to_use['gp_stats']:
                print(f"Generating groupby stats with cfg {gp_stats_cfg}...")
                keys = gp_stats_cfg['keys']
                feats, time_slots, shop_tags, stats = gp_stats_cfg['cfg']
                params = {'feats': feats, 'time_slots': time_slots, 
                          'shop_tags': shop_tags, 'stats': stats}
                if keys['cli_attrs'] == 'all':
                    keys_cli_attrs = CLI_ATTRS[1:]
                else: keys_cli_attrs = keys['cli_attrs']
                for cli_attr in keys_cli_attrs:
                    # Each client attribute is considered as key 
                    # (one per agg)
                    cli_attr = cli_attr if isinstance(cli_attr, list) \
                               else [cli_attr]
                    params['keys'] = {'cli_attrs': cli_attr,
                                      'apc_states': keys['apc_states']}
                    gp_keys, df_agg = fg.groupby_and_agg(**params)
                    if df_agg.size != 0:
                        self._dataset = self._dataset.merge(df_agg, 
                                                            on=gp_keys,
                                                            how='left')
                    del gp_keys, df_agg
            self._dataset.index = CHIDS
            self._dataset.index.name = 'chid'
            self._dataset.fillna(self._dataset.mean(), inplace=True)

            if not feats_to_use['use_cli_attrs']:
                self._dataset.drop(CLI_ATTRS[1:], axis=1, inplace=True)
            
        # Add groundtruths correponding to X samples into dataset
        if self._have_y:
            self._add_gts()
        
        # Randomly sample from entire dataset to speedup training 
        if self._rand_samples is not None:
            frac, rs = self._rand_samples[0], self._rand_samples[1]
            self._dataset = self._dataset.sample(frac=frac, random_state=rs)
        
        self.pk = self._dataset.index   # Primary key for predicting report 
        self._dataset.reset_index(inplace=True)
        
        # Drop disabled categorical features (must run through this cleaning
        # regardless of whether to use categorical or not, because there's
        # a need to clean the cat features (e.g., those from pk) in ._dataset.
        self._drop_cat(feats_to_use['use_chid'],
                       feats_to_use['chid_as_cat'],
                       feats_to_use['use_shop_tag'])
        
        if feats_to_use['use_cat']:
            # Preprocess categorical features to alleviate memory consumption of 
            # the training process using gbdt models (e.g., lgbm)
            self._proc_cat()
        else:
            self.cat_features_ = []
        
        # Select features either manually or automatically to avoid the
        # risk of overfitting caused by curse of dimensionality
        
        # Record all feature names
        self.features_ = [col for col in self._dataset if col != 'make_txn']
        
    def get_X_y(self):
        X_cols = self.features_
        X = self._dataset[X_cols]
        y = None if not self._have_y else self._dataset['make_txn'] 
        
        return X, y
    
    def _setup(self):
        '''Setup basic configuration.
        '''
        self._t_start = self._t_end - self._t_window + 1
        self._t_range = (self._t_start, self._t_end)
        self._pred_month = self._t_end + self._horizon
        if self._production:
            pass
        
        # Setup attibutes
        self.features_ = []
        self.cat_features_ = CAT_FEATURES.copy()   # Copy to avoid messing up 
                                                   # constant metadata access
                                                   # (notice removing of feat)
    
    def _get_raw_n(self, feats):
        '''Return raw numeric features without aggregation given the 
        lookback time window.
        *Note: If multi-class classification is specified, then only 
               return the DataFrame with unique `chid`s as indices.
        
        Parameters:
            feats: list, features to use
        
        Return:
            X_raw_n: pd.DataFrame, raw numeric features
        '''
        if self._mcls:
            # If the task is modelled as multi-class classification
            if self._production:
                # All chids must exist in production scheme, and if y exists,
                # #chids shrinks to align with chids in y.
                if feats is not None:
                    X_raw_n = fe.get_raw_n_mcls(feats, self._t_end)
                else:
                    X_raw_n = pd.DataFrame(CHIDS, columns=['chid'])
            else:
                # #chids reduces due to X set. To use all clients' txns in
                # predicting month, please use production scheme.
                X_raw_n = pd.read_parquet("./data/raw/raw_data.parquet",
                                          columns=['chid', 'dt'])
                X_raw_n = X_raw_n[X_raw_n['dt'] == self._t_end]
                X_raw_n.drop('dt', axis=1, inplace=True)
                X_raw_n.drop_duplicates(inplace=True, ignore_index=True)
            X_raw_n.set_index(keys=['chid'], drop=True, inplace=True)
        else:
            feats = PK + feats
            X_raw_n = fe.get_raw_n(feats, self._t_range, self._train_leg,
                                   self._production)
        
        return X_raw_n
    
    def _get_cli_attrs(self):
        '''Return client attribute vector for each client in current 
        month; that is, client attributes at dt=t_end
        
        Parameters:
            None
        
        Return:
            X_cli_attrs: pd.DataFrame, client attrs in current month
        '''
        feats = ['dt'] + CLI_ATTRS
        X_cli_attrs = fe.get_cli_attrs(feats, self._t_end, self._production)
        
        return X_cli_attrs
    
    def _get_tifu_vecs(self, params):
        '''Return client or predicting vectors based on the concept of 
        TIFU-KNN. For more detailed information, please refer to:
        Modeling Personalized Item Frequency Information for 
        Next-basket Recommendation.
        
        To boost fe efficiency, pre-computed tifu vectors can be dumped 
        and loaded here. Then, there's no re-computation overhead.
        
        Parameters:
            params: dict, hyperparemeters of TIFU-KNN
        
        Return:
            tifu_vecs: pd.DataFrame, client or predicting vector for 
                       each client containing either only legitimate 
                       shop_tags or all
        '''
        # Get client vector representation for each client
        purch_map_path = "./data/processed/purch_maps.pkl"
        t_lower_bound = self._t_end - params['t_window'] + 1 
        cli_vecs = fe.get_cli_vecs(purch_map_path=purch_map_path,
                                   t1=t_lower_bound, 
                                   t2=self._t_end, 
                                   gp_size=params['gp_size'],
                                   decay_wt_g=params['decay_wt_g'], 
                                   decay_wt_b=params['decay_wt_b'])
        
        if params['scale'] == 'cli':
            tifu_vecs = cli_vecs
        elif params['scale'] == 'pred':
            pred_vecs = fe.get_pred_vecs(cli_vecs=cli_vecs, 
                                         n_neighbor_candidates=params[
                                             'n_neighbor_candidates'
                                         ],
                                         sim_measure=params['sim_measure'],
                                         k=params['k'],
                                         alpha=params['alpha'])
            tifu_vecs = pred_vecs
        
        tifu_vecs = pd.DataFrame.from_dict(tifu_vecs, orient='index')
        if params['shop_tag_cstr'] == 'leg':
            # Only dimensions corresponding to legitimate shop_tags in tifu 
            # vectors will be retained
            tifu_vecs = tifu_vecs.iloc[:, LEG_SHOP_TAGS_INDICES]
        elif params['shop_tag_cstr'] == 'illeg':
            tifu_vecs = tifu_vecs.iloc[:, ILLEG_SHOP_TAGS_INDICES]
        tifu_vecs.columns = [f'tifu_shop_tag{i+1}_' for i in tifu_vecs.columns]
    
        return tifu_vecs
    
    def _get_feat_vecs(self, params, feat):
        '''Return feature vectors fusing historical information based 
        on the concept of TIFU-KNN.
        
        Parameters:
            params: dict, hyperparemeters of feature vector generation
            feat: str, feature name
        
        Return:
            feat_vecs: pd.DataFrame, client or predicting vector for 
                       each client containing either only legitimate 
                       shop_tags or all
        '''
        # Get client vector representation for each client
        if ('txn_amt' in feat) and ('pct' not in feat):
            feat_map_path = f"./data/processed/feat_map_txn_amt/{feat}.npz"
        else:    
            feat_map_path = f"./data/processed/feat_map/{feat}.npz"
        t_lower_bound = self._t_end - params['t_window'] + 1 
        feat_vecs = fe.get_feat_vecs(feat_map_path=feat_map_path,
                                     t1=t_lower_bound, 
                                     t2=self._t_end, 
                                     gp_size=params['gp_size'],
                                     decay_wt_g=params['decay_wt_g'], 
                                     decay_wt_b=params['decay_wt_b'])
        
        if params['scale'] == 'cli':
            feat_vecs = feat_vecs
        elif params['scale'] == 'pred':
            if params['sim_deter'] == 'cli_attr':
                print("Feat pred vector with sim determinant cli_attrs!")
                X_cli_attrs = fe.get_cli_attrs(['dt']+CLI_ATTRS, 
                                               self._t_end, 
                                               self._production)
            else: X_cli_attrs = None
            pred_vecs = fe.get_feat_pred_vecs(feat_vecs=feat_vecs, 
                                              n_neighbor_candidates=params[
                                                  'n_neighbor_candidates'
                                              ],
                                              sim_measure=params['sim_measure'],
                                              k=params['k'],
                                              alpha=params['alpha'],
                                              cli_attr_map=X_cli_attrs)
            feat_vecs = pred_vecs
        feat_vecs = pd.DataFrame.from_dict(feat_vecs, orient='index')
        
        if params['shop_tag_slctn'] != []:
            # Manual shop_tag selection is enabled
            idx_selected = np.array(params['shop_tag_slctn']) - 1
            feat_vecs = feat_vecs.iloc[:, idx_selected]
        elif params['shop_tag_cstr'] == 'leg':
            # Only dimensions corresponding to legitimate shop_tags in tifu 
            # vectors will be retained
            feat_vecs = feat_vecs.iloc[:, LEG_SHOP_TAGS_INDICES]
        elif params['shop_tag_cstr'] == 'illeg':
            feat_vecs = feat_vecs.iloc[:, ILLEG_SHOP_TAGS_INDICES] 
        feat_vecs.columns = [f'{feat}_shop_tag{i+1}_' for i 
                             in feat_vecs.columns]
    
        return feat_vecs
    
    def _post_proc_fp_mat(self, feat_pred_mat, fp_pproc):
        '''Post process the feature prediction matrix and retain only 
        specified raw feature vectors.
        
        Only support the specification that all feature prediction 
        vectors of the same feature are derived usin leg shop_tags.
        
        Parameters:
            feat_pred_mat: pd.DataFrame, raw feature prediction matrix
            fp_pproc: dict, configuration of post-processing
        
        Return:
            feat_pred_mat_: pd.DataFrame, post-processed feature matrix
                            with derived stats
        '''
        feat_pred_mat_ = feat_pred_mat.copy()
        for feat, pproc in fp_pproc.items():
            for shop_tag in LEG_SHOP_TAGS:
                feat_suffix = f'shop_tag{shop_tag}_'
                cols = [col for col in feat_pred_mat_.columns if 
                        col.startswith(feat) and (feat_suffix in col)]
                for stats in pproc['stats']:
                    if stats == 'mean':
                        stats_series = feat_pred_mat_[cols].std(axis=1)
                    elif stats == 'median':
                        stats_series = feat_pred_mat_[cols].skew(axis=1)
                    elif stats == 'std':
                        stats_series = feat_pred_mat_[cols].std(axis=1)
                    elif stats == 'skew':
                        stats_series = feat_pred_mat_[cols].skew(axis=1)
                    elif stats == 'kurt':
                        stats_series = feat_pred_mat_[cols].kurt(axis=1)
                    feat_pred_mat_[f'{cols[0]}{stats}'] = stats_series
                    del stats_series
                cols_to_drop = [cols[i] for i in pproc['raw_vecs_to_drop']]
                if cols_to_drop != []:
                    feat_pred_mat_.drop(cols_to_drop, axis=1, inplace=True)
        
        return feat_pred_mat_
    
    def _get_txn_related_feats(self, feat, shop_tag_cstr):
        '''Return feature vectors or matrices containing information
        about transaction behavior.
        
        To boost the efficiency of feature generations, legitimate
        shop_tag filtering is done in each feature generation utility
        function defined in `fe.py`.
        
        Parameters:
            feat: str, feature name
            shop_tag_cstr: str, shop_tag subset specification, the
                           choices are as follows:
                               {'leg', 'illeg'}
            
        Return:
            txn_feat_vecs: pd.DataFrame, feature vector of information
                           related to transaction behavior for each 
                           client containing either only legitimate 
                           shop_tags or all 
        '''
        txn_feat_vecs = fe.get_txn_related_feat(self._t_end, 
                                                feat, 
                                                shop_tag_cstr)
        txn_feat_vecs = pd.DataFrame.from_dict(txn_feat_vecs, orient='index')
        
        # Add feature names 
        cols = []
        if shop_tag_cstr == 'leg':
            shop_tags = LEG_SHOP_TAGS 
        elif shop_tag_cstr =='illeg':
            shop_tags = ILLEG_SHOP_TAGS
        else: shop_tags = SHOP_TAGS_
        if feat == 'st_tgl':
            st_suffix = ['00', '01', '10', '11']
            for shop_tag in shop_tags:
                cols_shop_tag = [f'{feat}_{st}_shop_tag{shop_tag}' for st
                                 in st_suffix]
                cols += cols_shop_tag
        elif feat == 'n_shop_tags':
            months_hard_coded = range(self._t_end-6, self._t_end)
            cols = [f'{feat}_dt_{month+1}' for month in months_hard_coded]
        else:
            cols = [f'{feat}_shop_tag{i}' for i in shop_tags]
        txn_feat_vecs.columns = cols
        
        return txn_feat_vecs
    
    def _add_gts(self):
        '''Add y labels corresponding to X samples into dataset.
        
        Parameters:
            None
        
        Return:
            None
        '''
        y = pd.read_parquet("./data/raw/raw_data.parquet", columns=PK)
        chids_leg = self._get_leg_chids(y)
        y = y[y['dt'] == self._pred_month]
        y.drop('dt', axis=1, inplace=True)
        
        if self._mcls:
            y = y[y['chid'].isin(chids_leg)]
            y.set_index(keys=['chid'], drop=True, inplace=True)
            if self._train_leg:
                # shop_tags are presented in y in multi-class case, that's why
                # shop_tag specification isn't implemented in _get_raw_n().
                y = y[y['shop_tag'].isin(LEG_SHOP_TAGS)]
                y['shop_tag'] = y['shop_tag'].replace(LEG_SHOP_TAG_MAP)
            else:
                y['shop_tag'] = y['shop_tag'] - 1
            y.columns = ['make_txn']   # y values are indices of shop_tags
                                       # (i.e., orig_shop_tag - 1)
            self._dataset = self._dataset.join(y, how='right')
            # Dropna never takes effect if in production scheme, because all 
            # chids are processed in the X generation stage.
            self._dataset.dropna(inplace=True) 
        else:
            y.set_index(keys=['chid', 'shop_tag'], drop=True, inplace=True)
            y['make_txn'] = 1   # Assign 1 for transactions made in the month 
                                # we want to predict

            # Add groundtruths by joining with the transaction records of the 
            # month we want to predict
            self._dataset = self._dataset.join(y, how='left')
            self._dataset.fillna(0, inplace=True)   # Assign 0 for shop_tags 
                                                    # not bought by each cli. 
        self._dataset['make_txn'] = self._dataset['make_txn'].astype(np.int8)
    
    def _get_leg_chids(self, y):
        '''Return legitimate `chid`s allowed to appear in y dataset. 
        
        Parameters:
            y: pd.DataFrame, raw data containing all pks
        
        Return:
            chids_leg: list, legitimate `chid`s allowed to appear in y
        '''
        # Initialize legitimate chids to all chids
        chids_leg = list(y['chid'].unique())
        
        # Filter legitimate chids using client cold-start filter
        if self._drop_cold_start_cli:
            print(f"Finding and dropping clients with cold-start issue...")
            chids_leg = self._get_non_cold_start_chids(y)
        
        # Filter legitimate chids using zero-ndcg chid filter
        if self._drop_zero_ndcg_cli:
            print(f"Dropping clients with Zero-NDCG on training set evaluated"
                  " using top1 and top2 models...")
            chids_0ndcg = []
            with open("./chids_0ndcg_in_both_38_30_tr.txt", 'r') as f:
                for chid in f.readlines():
                    chids_0ndcg.append(int(chid.strip()))
            chids_leg = list(set(chids_leg).difference(set(chids_0ndcg)))
        
        return chids_leg
        
    def _get_non_cold_start_chids(self, y):
        '''Return legitimate `chid`s that aren't cold-start clients 
        joining in pred_month. Also, those joining at large dts are 
        filtered out to improve the feature quality.
        
        Parameters:
            y: pd.DataFrame, raw data containing all pks
            
        Return:
            chids_leg: list, legitimate `chid`s of non cold-start cli.s
        '''
        # Configure the illegal dt interval that clients join
        dt_join_tolerance = self._t_end - self._gen_feat_tolerance 
        dts_join_illeg = range(dt_join_tolerance+1, self._pred_month+1)
        dts_join_illeg = [dt for dt in dts_join_illeg]
        
        # Extract chids with dts in legitimate dt interval
        y_ = y.copy()
        y_ = y_[y_['shop_tag'].isin(LEG_SHOP_TAGS)]   # Flexible for illeg?
        dts_join = y_.groupby(by=['chid'])['dt'].min()
        dts_join_leg = dts_join[~dts_join.isin(dts_join_illeg)]
        
        # Extract legitimate chids present in pred_month
        y_pred_month = y_[y_['dt'] == self._pred_month]
        chids_pred_month = list(y_pred_month['chid'].unique())
        chids_leg = dts_join_leg[dts_join_leg.index.isin(chids_pred_month)]
        chids_leg = list(chids_leg.index)
        
        return chids_leg
        
    def _drop_cat(self, use_chid, chid_as_cat, use_shop_tag):
        '''Drop disabled categorical features.
        
        Parameters:
            use_chid: bool, whether chid is used 
            chid_as_cat: bool, whether chid is treated as categorical data
            use_shop_tag: bool, whether shop_tag is used
        '''
        self.cat_features_.remove('dt')
        self.cat_features_.remove('primary_card')
        if not use_chid:
            self._dataset.drop('chid', axis=1, inplace=True)
            self.cat_features_.remove('chid')
        elif not chid_as_cat:
            # Treat chid as a numeric feature
            self.cat_features_.remove('chid')
            
        if not use_shop_tag:
            try:
                self._dataset.drop('shop_tag', axis=1, inplace=True)
            except:
                # For the case that `shop_tag` isn't used as features, (e.g.,
                # multi-class classification),
                print("shop_tag doesn't exist in dataset!")
            self.cat_features_.remove('shop_tag')
            
    def _proc_cat(self): 
        '''Preprocess categorical features to alleviate the memory load
        for training process using gbdt models (e.g., lgbm).
        
        The main purpose is to make categories a list of continuous
        integers starting from 0. For more detailed information, please 
        refer to:
            https://lightgbm.readthedocs.io/en/latest/Quick-Start.html
            
        Parameters:
            None
            
        Return:
            None
        '''
        for cat_feat in self.cat_features_:
            if cat_feat == 'poscd':
                self._dataset['poscd'] = (self._dataset['poscd']
                                              .replace(99, 11))
            elif cat_feat == 'cuorg':
                self._dataset['cuorg'] = (self._dataset['cuorg']
                                              .replace([35, 38, 40], 
                                                       [10, 33, 34]))
            elif CAT_FEAT_LBOUNDS[cat_feat] == 0:
                pass
            else:
                self._dataset[cat_feat] = (self._dataset[cat_feat] - 
                                           CAT_FEAT_LBOUNDS[cat_feat])
                
            # Convert dtypes to shrink down memory consumption and 
            # also follow the advanced topics introduced in lgbm 
            # document accessible in doc string above
            if cat_feat == 'chid':
                self._dataset['chid'] = (self._dataset['chid']
                                             .astype(np.int32))
            else:
                self._dataset[cat_feat] = (self._dataset[cat_feat]
                                                 .astype(np.int8))
