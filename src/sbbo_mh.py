import numpy as np
import pandas as pd

from scipy.special import softmax
    

class SBBO:
    """
   Simulation based Bayesian Optimization

    Args:

        - 
    """

    def __init__(self, co_problem, model, cooling_schedule, af="avg", burnin=0.1):

        self.co_problem = co_problem
        self.af = af
        self.X = self.co_problem.X
        self.y = self.co_problem.y
        self.model = model
        self.model.fit(self.X, self.y)

        self.cooling_schedule = cooling_schedule
        self.burnin = burnin

        self.z_samples = np.zeros( [len(self.cooling_schedule),
                            self.co_problem.ncov] )
 

    def init_search(self):

        # z_init = self.Z_set[ np.random.choice(self.Z_set.shape[0]) ]
        z_init = self.co_problem.generate_candidate()
        z_init_d = self.co_problem.dummify(z_init)

        init_pred = self.model.pred_dist(z_init_d)
        y_sample = init_pred.sample(self.cooling_schedule[0])
       
        value = self.utility(y_sample, z_init, flag=self.af) # Watch out, parallel!
        value = np.sum(np.log(value))   
        value /= self.cooling_schedule[0]

        return z_init, y_sample, value

    def metropolis_step(self, H, value_old, y_sample_old, z_old):

        z_old = z_old.reshape(1,-1)
        idx = np.random.choice(np.arange(z_old.shape[1]))
        Z_candidates = self.co_problem.generate_candidates_idx(z_old, idx)
        Z_candidates_d = self.co_problem.dummify(Z_candidates)

        preds = self.model.pred_dist(Z_candidates_d)
        idx_new = np.random.choice( np.arange(Z_candidates.shape[0]) )
        z_new = Z_candidates[idx_new]
   
        y_sample = preds[idx_new].sample(H)

        value = self.utility(y_sample, z_new, flag=self.af) # Watch out, parallel!
        value = np.sum(np.log(value))   
        value /= H
        prob =  np.exp(H*value - H*value_old)
        
        if np.random.uniform() < prob:
            
            # print("Accept!")
            return z_new, y_sample, value
        
        else:
            
            return z_old, y_sample_old, value_old



    def update_all(self, temp, z_init, y_sample, value):

        # Update z
        for idx in range(self.co_problem.ncov):
            z_init, y_sample, value = self.metropolis_step(temp, value, y_sample, z_init)

        return z_init, y_sample, value


    def iterate(self):

        z_init, y_sample, value = self.init_search()

        for i, temp in enumerate(self.cooling_schedule):
            if i%1 == 0:
                print("Percentage completed:", 
                np.round( 100*i/len(self.cooling_schedule), 2) )
                print("Current state", z_init.reshape(5,-1))
                print("Current energy", self.model.predict(self.co_problem.dummify(z_init.reshape(1,-1)) ))
                print(np.mean(y_sample))

            z_init, y_sample, value = self.update_all(temp, z_init, y_sample, value)
            

                
        z_star, quality = self.extract_solution()
        
        return z_star, self.z_samples, quality



    def utility(self, y, z, flag='avg'):

        if flag == 'EI':

            result = np.zeros_like(y) + 0.0001
            result[y > self.y.max()] = y[y > self.y.max()]  - self.y.max()
            return result

        elif flag == 'avg':
            return y

            


    def get_mode(self, samples):
        vals, counts = np.unique( samples, return_counts=True, axis=0 )
        index = np.argmax(counts)
        return vals[index]

    # CHECK!
    def extract_solution(self):

        z_star = np.zeros_like(self.z_samples[0])
        burnin_end = int(self.burnin * len(self.cooling_schedule) )
        
        for j in range(z_star.shape[0]):
            z_star[j] = self.get_mode(self.z_samples[  burnin_end: , j ])

        # quality = self.co_problem.compute_obj(z_star)
        quality = self.model.predict(self.co_problem.dummify(z_star.reshape(1,-1)) )
        return z_star, quality





    
    