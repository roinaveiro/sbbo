import numpy as np
import pandas as pd
    

class SBBO:
    """
   Simulation based Bayesian Optimization

    Args:
        
    """

    def __init__(self,  l=5):
 

    def model():
        pass

    def predict():
        pass

    def sample():
        pass



        
    


    @staticmethod
    def desdummify(x, l):
        return np.where(x.reshape(-1, l) == 1)[1]

    @staticmethod
    def dummify(x, l):
        return np.eye(l)[x.astype(int)].flatten()


    
    