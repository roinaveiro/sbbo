import numpy as np
import pandas as pd

from scipy.special import softmax
    

class SBBO:
    """
   Simulation based Bayesian Optimization

    Args:

        - 
        
    """

    def __init__(self, co_problem, method, fname):

        self.co_problem = co_problem
        self.search_method = method

        self.X = self.search_method.X
        self.y = self.search_method.y

        self.fname = fname



    def get_candidate(self):
        candidate, _ = self.search_method.iterate()
        return candidate

    def evaluate_candidate(self, candidate):
        value = self.co_problem.compute_obj(candidate)
        value_unscaled = self.co_problem.compute_obj(candidate, scale=False)
        return value, value_unscaled
    
    def update(self, candidate, value):

        self.X = np.vstack([self.X, candidate])
        self.y = np.append(self.y, value)
        self.search_method.update(candidate, value)


    def iterate(self, n_eval):

        best_vals = np.zeros(n_eval)
        current_vals = np.zeros(n_eval)
        iters = np.arange(n_eval)


        for i in range(n_eval):
            candidate = self.get_candidate()
            value, value_unscaled =  self.evaluate_candidate(candidate)
            self.update(candidate, value)
            
            iters[i] = i
            best_vals[i] = self.co_problem.compute_obj(self.X[self.y.argmax()], scale=False)
            current_vals[i] = value_unscaled

            print("Iter:", i)
            print("#########################################")
            print("Current value: ", value_unscaled)
            print("Best value: ", best_vals[i])
            print("Best X", self.X[self.y.argmax()])

            print("Dim X", self.X.shape)
            print("Dim y", self.y.shape)
            print("#########################################")

        df = pd.DataFrame({"iter": iters,
                      "best_vals": best_vals,
                      "current_vals": current_vals,
                      })
        
        df.to_csv(self.fname, index=False)


    def extract_best(self):
        return self.X[self.y.argmax()]

   
