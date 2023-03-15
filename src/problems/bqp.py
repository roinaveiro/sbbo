import numpy as np
import pandas as pd

from src.config import ALPHA_BQP, LMB_BQP, NVARS_BQP

def _quad_mat(n_vars, alpha, seed):
 
    # evaluate decay function
    i = np.linspace(1,n_vars,n_vars)
    j = np.linspace(1,n_vars,n_vars)

    K = lambda s,t: np.exp(-1*(s-t)**2/alpha)
    decay = K(i[:,None], j[None,:])

    # Generate random quadratic model
    # and apply exponential decay to Q
    Q  = np.random.RandomState(seed).randn(n_vars, n_vars)
    Qa = Q*decay

    return Qa


class BQP(object):
    """
    Random Binary Quadratic Problem
    """
    def __init__(self, n=10, random_seed=305):
        self.lamda = LMB_BQP
        self.n_init = n
        self.ncov = NVARS_BQP
        
        self.Q = _quad_mat(self.ncov, ALPHA_BQP, random_seed)

        self.scaler = 5.0
       
        # In all evaluation, the same sampled values are used.
        self.generate_init_data(random_seed)

    def evaluate(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.array([self.compute_obj(x[i]) for i in range(x.shape[0])])

    def compute_obj(self, x, scale=True):
        if x.ndim == 2:
            x = x.squeeze()

        assert x.ndim == 1
        evaluation = np.sum(x.dot(self.Q)*x) + self.lamda*np.sum(x)

        if scale:
            return evaluation + self.scaler
        else:
            return evaluation
    
    def generate_init_data(self, seed):
        self.X = np.random.RandomState(seed).choice([0.0,1.0],
                                   size=(self.n_init, self.ncov))
        self.y = self.evaluate(self.X)


    def generate_candidate(self):
        return np.random.choice([0,1], size=self.ncov).reshape(1,-1)

    def generate_candidates_idx(self, z, idx):
        result = np.repeat(z, 2, axis=0)
        result[:, idx] = np.arange(2)
        return result
    
    def desdummify(self, x):
        return x

    def dummify(self, x):
        return x