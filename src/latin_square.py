import numpy as np
import pandas as pd
    

class LatinSquare:
    """
    Latin square problem of dimension l

    Args:
        l (int): Latin Square dimension
    """

    def __init__(self, n=40, l=5, std=0.01):
        self.l = l
        self.n_init = n
        self.std = std
        self.ncov = self.l**2
        self.max_obj = 2 * (self.l * (self.l - 1) ) + 3*self.std
        self.min_obj = -3 * self.std

        self.generate_init_data(self.n_init)


    def compute_obj(self, x, scale=True):

        x = self.desdummify(x)
        x = x.reshape(self.l, -1)
    
        rows = np.apply_along_axis(self.get_cost, axis=0, arr=x)
        cols = np.apply_along_axis(self.get_cost, axis=1, arr=x)

        tmp = np.sum(rows+cols) + np.random.normal(loc=0, scale=self.std)
        if scale:
            return (self.max_obj - tmp)/self.max_obj 
        
        else:
            return tmp

    def generate_init_data(self, n=100, dummies=True):

        self.X_nodum = np.random.randint(self.l, size=(n, self.l**2)).astype(float)
        if dummies:
            self.X = self.dummify(self.X_nodum)
            #df = pd.DataFrame(self.X_nodum)
            #self.X = pd.get_dummies(df.astype('str')).values
            #self.X = self.X.astype(float)

        self.y = np.zeros(n)
        for i in range(n):
            self.y[i] = self.compute_obj(self.X[i,:])

    def generate_candidate(self, n=1):
        cand_nodum = np.random.randint(self.l, size=(n, 
                                        self.l**2) ).astype(float)

        return cand_nodum

    def generate_candidates_idx(self, z, idx):
        result = np.repeat(z, 5, axis=0)
        result[:, idx] = np.arange(self.l).astype(float)
        return result
        
    
    def desdummify(self, x):
        return np.where(x.reshape(-1, self.l) == 1)[1]# .reshape(x.shape[0], -1)


    def dummify(self, x):
        return np.eye(self.l)[x.astype(int)].flatten().reshape(x.shape[0], -1)


    @staticmethod
    def get_cost(x):
        _, counts = np.unique(x, return_counts=True)
        return np.sum(counts-1)


    


    
    