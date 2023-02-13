import numpy as np
import pandas as pd
    

class LatinSquare:
    """
    Latin square problem of dimension l

    Args:
        l (int): Latin Square dimension
    """

    def __init__(self,  l=5):
        self.l = l
        self.generate_init_data()

    def compute_obj(self, x):

        x = self.desdummify(x, l=self.l)
        x = x.reshape(self.l, -1)
    
        rows = np.apply_along_axis(self.get_cost, axis=0, arr=x)
        cols = np.apply_along_axis(self.get_cost, axis=1, arr=x)
        
        return np.sum(rows+cols) + np.random.normal(loc=0, scale=0.1)

    def generate_init_data(self, n=100, dummies=True):

        self.X_nodum = np.random.randint(self.l, size=(n, self.l**2)).astype(float)
        if dummies:
            df = pd.DataFrame(self.X_nodum)
            self.X = pd.get_dummies(df.astype('str')).values
            self.X = self.X.astype(float)

        self.y = np.zeros(n)
        for i in range(n):
            self.y[i] = self.compute_obj(self.X[i,:])

        
    

    @staticmethod
    def get_cost(x):
        _, counts = np.unique(x, return_counts=True)
        return np.sum(counts-1)

    @staticmethod
    def desdummify(x, l):
        return np.where(x.reshape(-1, l) == 1)[1]

    @staticmethod
    def dummify(x, l):
        return np.eye(l)[x.astype(int)].flatten()


    
    