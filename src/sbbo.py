import numpy as np
import pandas as pd

from scipy.special import softmax
from ngboost import NGBRegressor
    

class SBBO:
    """
   Simulation based Bayesian Optimization

    Args:

        - 
        
    """

    def __init__(self, co_problem, model, cooling_schedule, burnin=0.1):

        self.co_problem = co_problem
        self.X = self.co_problem.X
        self.y = self.co_problem.y
        self.model = model
        self.model.fit(self.X, self.y)

        self.cooling_schedule = cooling_schedule
        self.burnin = burnin

        self.z_samples = np.zeros( [len(self.cooling_schedule),
                            self.co_problem.ncov] )
 

    def init_search(self):

        ##
        hmms = []
        value = 0

        # z_init = self.Z_set[ np.random.choice(self.Z_set.shape[0]) ]
        z_init = self.co_problem.generate_candidate()
        z_init_d = self.co_problem.dummify(z_init)

        for temp in range(self.cooling_schedule[0]):

            init_pred = self.model.pred_dist(z_init_d)
            y_sample = init_pred.sample(self.cooling_schedule[0])
            # value += np.log( self.attacker.utility(z_init, hmm_sample) )


        return z_init, z_init_d, y_sample

    def update_probs_metropolis(self, H, value_old, y_sample_old, z):

        z_d = self.co_problem.dummify(z)
        value = 0
        pred = self.model.pred_dist(z_d)
        y_sample = pred[0].sample(H)

        value = self.utility(y_sample, z_d, flag='EI') # Watch out, parallel!
        value = np.sum(np.log(value))   
        value /= H
        prob =  np.exp(H*value - H*value_old)
        
        if np.random.uniform() < prob:
            
            #print("Accept!")
            return y_sample, value
        
        else:
            
            return y_sample_old, value_old

    def update_z(self, z, idx, y_samples):

        Z_candidates = self.generate_candidates_idx(z, idx)
        Z_candidates_d = self.co_problem.dummify(Z_candidates)

        preds = self.model.pred_dist(Z_candidates_d)
        energies = np.zeros( len(Z_candidates) )

        for i in range(Z_candidates.shape[0]):

            z_new = Z_candidates[i,:]

            for y in y_samples:
                 
                energies[i] += (np.log( self.utility(y, z_new) ) + 
                                preds[i].dist.logpdf(y))
                
        p = softmax(energies)
        #print(p)

        candidate_idx = np.random.choice( 
            np.arange(len(Z_candidates) ), p=p )
        
        return Z_candidates[candidate_idx]


    def update_all(self, i, temp, z_init, y_sample, value):

        # Update z
        for idx in range(self.co_problem.ncov):
            z_init = self.update_z(z_init, idx, y_sample)
            self.z_samples[i, idx] = z_init[idx]

        # Update value with new z 
        pass
        

        y_sample, value = self.update_probs_metropolis(temp, value,
                                                         y_sample, z_init)

        return z_init, y_sample, value


    def utility(self, y, z, flag='EI'):

        if flag == 'EI':
            if y > self.y.max():
                return y - self.y.max()
            else:
                return 0.0





    
    