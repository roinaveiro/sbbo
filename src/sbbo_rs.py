import numpy as np
import pandas as pd
import copy

class RS:
    """
   Simulation based Bayesian Optimization - RS

    Args:

        - 
    """

    def __init__(self, co_problem, params):
        self.co_problem = co_problem
        self.X = self.co_problem.X
        self.y = self.co_problem.y
        
    def iterate(self):
        return self.co_problem.generate_candidate(), None
    
    def update(self, candidate, value):
        self.X = np.vstack([self.X, candidate])
        self.y = np.append(self.y, value)
    

    
    
