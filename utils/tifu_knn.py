'''
TIFU-KNN vector generator.
Author: JiaWei Jiang

This file implements feature aggregation novelty proposed in TIFU-KNN,
trying to fusing the historical information of original purchasing map;
furthermore, nearest neightbors' purchasing behaviors are considered.
'''
# Import packages 
import os 
import math
from tqdm import tqdm
import pickle

import numpy as np

from metadata import *

# TIFU-KNN
class CliVecGenerator:
    '''Generate client vector representation for a single client based 
    on concept of TIFU-KNN. 
    
    Parameters:
        purch_map_path: str, path to pre-dumped purchasing maps
        t1: int, time lower bound 
        t2: int, time upper bound (exclusive)
        gp_size: int, size of each purchasing submap
        decay_wt_g: float, weight decay ratio across neighbor groups
        decay_wt_b: float, weight decay ratio within each group
    '''
    def __init__(self, purch_map_path, t1, t2, 
                 gp_size, decay_wt_g, decay_wt_b):
        with open(purch_map_path, 'rb') as f:
            self.purch_maps = pickle.load(f)
        self.t1 = t1
        self.t2 = t2  
        self.gp_size = gp_size
        self.decay_wt_g = decay_wt_g
        self.decay_wt_b = decay_wt_b
        self._setup()
        
    def get_client_vec(self, chid):
        '''Return the client vector represented by fusing repeated purchase
        pattern and collaborative one.

        Parameters:
            chid: int, client identifier

        Return:
            client_vec: ndarray, client vector representation
        '''
        purch_map = self.purch_maps[chid][self.t1:self.t2]
        if self.first_gp_size != 0:
            first_gp = purch_map[:self.first_gp_size]
            first_gp = first_gp * self.wt_g[0]
            first_gp = np.einsum('ij, i->j', first_gp, self.wt_b[self.first_gp_size:])
            
        normal_gps = np.reshape(purch_map[self.first_gp_size:], 
                                self.normal_gp_shape)   
        normal_gps = np.einsum('ijk, i->jk', normal_gps, self.normal_gp_wt)
        normal_gps = np.einsum('ij, i->j', normal_gps, self.wt_b)
        client_vec = normal_gps if self.first_gp_size == 0 else first_gp + normal_gps
    
        return client_vec
    
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
            
def get_cli_vecs(purch_map_path, t1, t2, 
                 gp_size, decay_wt_g, decay_wt_b):
    '''Return client vector representation for each client.
    
    Parameters:
        purch_map_path: str, path to pre-dumped purchasing maps
        t1: int, time lower bound 
        t2: int, time upper bound (exclusive)
        gp_size: int, size of each purchasing submap
        decay_wt_g: float, weight decay ratio across neighbor groups
        decay_wt_b: float, weight decay ratio within each group
    Return:
        cli_vecs: dict, client vector representation for each client
    '''
    cli_vec_generator = CliVecGenerator(purch_map_path=purch_map_path, 
                                        t1=t1, 
                                        t2=t2, 
                                        gp_size=gp_size, 
                                        decay_wt_g=decay_wt_g, 
                                        decay_wt_b=decay_wt_b)
    cli_vecs = {}
    for chid in tqdm(cli_vec_generator.purch_maps.keys()):
        cli_vecs[chid] = cli_vec_generator.get_client_vec(chid)
    
    return cli_vecs

def get_pred_vecs(cli_vecs, n_neighbor_candidates, sim_measure, k, alpha):
    '''Return client prediction vector representation for each client
    considering both repeated (client-specific) and collaborative
    purchasing patterns.
    
    Parameters:
        cli_vecs: dict, client vector representation for each client
        n_neighbor_candidates: int, number of neighboring candidates 
                               used to do similiarity measurement
        sim_measure: str, similarity measure criterion 
        k: int, number of nearest neighbors
        alpha: float, balance between client-specific and collaborative
               patterns
    '''
    pred = {}
    cli_map = np.array([v for v in cli_vecs.values()])
    
    for chid, target_vec in tqdm(cli_vecs.items()):
        sim_map = {}
        un = np.zeros(N_SHOP_TAGS)
        neighbor_candidates = sample(range(N_CLIENTS), n_neighbor_candidates)
        neighbor_mat = cli_map[neighbor_candidates]
        
        if sim_measure == 'cos':
            dot_sim = np.matmul(neighbor_mat, target_vec)
            target_norm = np.linalg.norm(target_vec)
            neighbor_norm = np.linalg.norm(neighbor_mat, axis=1)
            sim_vec = dot_sim / (target_norm * neighbor_norm) 
        elif sim_measure == 'ed':
            vec_sub = neighbor_mat - target_vec
            sim_vec = np.linalg.norm(vec_sub, axis=1)
        
        sim_map = {chid_: sim for chid_, sim in zip(neighbor_candidates, sim_vec)}
        sim_map = dict(sorted(sim_map.items(), 
                              key=lambda item: item[1], 
                              reverse=True))
        neighbors = list(sim_map.keys())[:k]
        
        for n in neighbors:
            un += cli_vecs[n+int(1e7)]
        un = un / k
        pred[chid] = alpha*target_vec + (1-alpha)*un
        
        del sim_map, un, neighbor_candidates, neighbor_mat, neighbors
    
    return pred