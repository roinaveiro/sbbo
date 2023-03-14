import numpy as np
import pandas as pd

from seqfold import dg
from src.config import RNA_CHAIN_LEN



class RNA(object):
    """
    RNA MFE optimization problem
    """
    def __init__(self,  n=10, random_seed=305):
        self.n_init = n
        self.ncov = RNA_CHAIN_LEN
        self.letters = 4
        self.dict = {0:"A", 1:"C", 2:"G", 3:"T"}

        self.scaler = 10.0

        self.generate_init_data(random_seed)

    def evaluate(self, x):
        return np.array([self.compute_obj(x[i]) for i in range(x.shape[0])])

    def compute_obj(self, x, scale=True):
        x = x.reshape(1,-1)
        mfe = dg(self.to_seq(x)[0], temp = 37.0)
        if scale:
            return -mfe + self.scaler
        else:
            return mfe
    
    def generate_init_data(self, seed, dummies=True):
        X_nd = np.random.RandomState(seed).randint(self.letters,
                                                      size=(self.n_init, self.ncov)).astype(float)
        
        if dummies:
            self.X = self.dummify(X_nd)
        else:
            self.X = X_nd

        self.y = self.evaluate(X_nd)


    def generate_candidate(self, n=1):
        return  np.random.randint(self.letters, size=(n, self.ncov))

    def generate_candidates_idx(self, z, idx):
        pass
    
    def desdummify(self, x, n):
        aux = np.arange(1, self.letters).reshape(-1,1)
        return np.dot(x.reshape(n, -1, self.letters - 1), aux).reshape(n,-1)

    def dummify(self, x):
        aux = np.vstack([np.zeros(self.letters-1), np.eye(self.letters-1)])
        return aux[x.astype(int)].flatten().reshape(x.shape[0], -1)
    
    def to_seq(self, X):
        tmp = np.vectorize(self.dict.get)(X)
        return [''.join([''.join(x) for x in tmp[i]]) for i in np.arange(tmp.shape[0])]



    


    
    
