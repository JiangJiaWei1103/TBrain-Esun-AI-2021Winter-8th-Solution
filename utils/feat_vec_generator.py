'''
Feature vector generator.
Author: JiaWei Jiang

This file extends the feat aggregation technique proposed in TIFU-KNN,
trying to fusing the historical information of original feature map. 
Furthermore, neighboring (spatial) effects are also considered.fea
'''
# Import packages 
import os 
import math
from tqdm import tqdm
from random import sample

import numpy as np

from metadata import *

class FeatVecGenerator:
    '''Generate feature vector representation of any feature map for a 
    single client based on concept of TIFU-KNN.
    
    Parameters:
        feat_map_path: str, path to pre-dumped feature maps
        t1: int, time lower bound 
        t2: int, time upper bound (exclusive)
        gp_size: int, size of each feature submap
        decay_wt_g: float, weight decay ratio across neighbor groups
        decay_wt_b: float, weight decay ratio within each group
    '''
    def __init__(self, feat_map_path, t1, t2, 
                 gp_size, decay_wt_g, decay_wt_b):
        self.feat_maps = np.load(feat_map_path)['arr_0']
        self.t1 = t1
        self.t2 = t2  
        self.gp_size = gp_size
        self.decay_wt_g = decay_wt_g
        self.decay_wt_b = decay_wt_b
        self._setup()
        
    def get_feat_vec(self, chid):
        '''Return the feature vector fusing the historical (temporal)
        information from the specified feature map.

        Parameters:
            chid: int, client identifier

        Return:
            feat_vec: ndarray, feature vector representation
        '''
        feat_map = self.feat_maps[int(chid-1e7), ...][self.t1:self.t2]
        if self.first_gp_size != 0:
            first_gp = feat_map[:self.first_gp_size]
            first_gp = first_gp * self.wt_g[0]
            first_gp = np.einsum('ij, i->j', 
                                 first_gp, self.wt_b[-self.first_gp_size:])
            
        normal_gps = np.reshape(feat_map[self.first_gp_size:], 
                                self.normal_gp_shape)   
        normal_gps = np.einsum('ijk, i->jk', normal_gps, self.normal_gp_wt)
        normal_gps = np.einsum('ij, i->j', normal_gps, self.wt_b)
        feat_vec = normal_gps if self.first_gp_size == 0 else first_gp + normal_gps
    
        return feat_vec
    
    def _setup(self):
        self.n_baskets = self.t2 - self.t1   # See one month as one basket
                                             # time interval is like [t1, t2)
        self.n_gps = math.ceil(self.n_baskets / self.gp_size)
        self.wt_g = [pow(self.decay_wt_g, p) for p in range(self.n_gps-1, -1, -1)]
        self.wt_b = [pow(self.decay_wt_b, p) for p in range(self.gp_size-1, -1, -1)]
        
        self.first_gp_size = self.n_baskets % self.gp_size
        if self.first_gp_size == 0:
            # If each group has the same size
            self.normal_gp_shape = (self.n_gps, self.gp_size, -1)
            self.normal_gp_wt = self.wt_g
        else:
            self.normal_gp_shape = (self.n_gps-1, self.gp_size, -1)   # Ignore the first gp
            self.normal_gp_wt = self.wt_g[1:]   # Ignore the first gp
            
def get_feat_vecs(feat_map_path, t1, t2, gp_size, 
                  decay_wt_g, decay_wt_b):
    '''Return feature vector representation for each client.
    
    Parameters:
        feat_map_path: str, path to pre-dumped feature maps
        t1: int, time lower bound 
        t2: int, time upper bound (exclusive)
        gp_size: int, size of each purchasing submap
        decay_wt_g: float, weight decay ratio across neighbor groups
        decay_wt_b: float, weight decay ratio within each group
    Return:
        feat_vecs: dict, feature vector representation for each client
    '''
    feat_vec_generator = FeatVecGenerator(feat_map_path=feat_map_path, 
                                          t1=t1, 
                                          t2=t2, 
                                          gp_size=gp_size, 
                                          decay_wt_g=decay_wt_g, 
                                          decay_wt_b=decay_wt_b)
    feat_vecs = {}
    for chid in tqdm(CHIDS):
        feat_vecs[chid] = feat_vec_generator.get_feat_vec(chid)
    
    return feat_vecs

def get_feat_pred_vecs(feat_vecs, n_neighbor_candidates, sim_measure, k, 
                       alpha, cli_attr_map=None):
    '''Return feature prediction vector representation for each client
    considering both repeated (client-specific) and collaborative
    (neightboring) feature patterns.
    
    Parameters:
        feat_vecs: dict, feature vector representation for each client
        n_neighbor_candidates: int, number of neighboring candidates 
                               used to do similiarity measurement
        sim_deter: str, elements used to compute similarity (e.g.,
                   client attribute, feature vector)
        sim_measure: str, similarity measure criterion 
        k: int, number of nearest neighbors
        alpha: float, balance between client-specific and collaborative
               patterns
        cli_attr_map: pd.DataFrame, client attribute for each client
    
    Return:
        pred_vecs: dict, predicting feature vector for each client
    '''
    pred_vecs = {}
    feat_map = np.array([v for v in feat_vecs.values()])
    if cli_attr_map is not None:
        use_cli_attr_as_sim_deter = True
        cli_attr_map = cli_attr_map.values
    else: use_cli_attr_as_sim_deter = False
    
    for i, (chid, target_vec) in tqdm(enumerate(feat_vecs.items())):
        sim_map = {}
        un = np.zeros(N_SHOP_TAGS)
        neighbor_candidates = sample(range(N_CLIENTS), n_neighbor_candidates)
        if use_cli_attr_as_sim_deter:
            neighbor_mat = cli_attr_map[neighbor_candidates]
            sim_vec_chid = cli_attr_map[i]
            sim_vec = np.all(neighbor_mat == sim_vec_chid, axis=1)
            neighbors = neighbor_candidates * sim_vec
            neighbors = neighbors[neighbors != 0]
        else: 
            neighbor_mat = feat_map[neighbor_candidates]
            sim_vec_chid = target_vec
            if sim_measure == 'cos':
                dot_sim = np.matmul(neighbor_mat, sim_vec_chid)
                target_norm = np.linalg.norm(sim_vec_chid)
                neighbor_norm = np.linalg.norm(neighbor_mat, axis=1)
                sim_vec = dot_sim / (target_norm * neighbor_norm) 
            elif sim_measure == 'ed':
                vec_sub = neighbor_mat - sim_vec_chid
                sim_vec = np.linalg.norm(vec_sub, axis=1)
        
            sim_map = {chid_: sim for chid_, sim in zip(neighbor_candidates, sim_vec)}
            sim_map = dict(sorted(sim_map.items(), 
                                  key=lambda item: item[1], 
                                  reverse=True))
            neighbors = list(sim_map.keys())[:k]

        for n in neighbors:
            un += feat_vecs[n+int(1e7)] / np.max(feat_vecs[n+int(1e7)])   # Normalize to avoid dominant client???
        un = un / len(neighbors)
        pred_vecs[chid] = np.nansum([alpha*target_vec / np.max(target_vec), 
                                    (1-alpha)*un], axis=0)
        
        del sim_map, un, neighbor_candidates, neighbor_mat, \
            sim_vec_chid, sim_vec, neighbors
    
    return pred_vecs