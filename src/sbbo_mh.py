import numpy as np
import pandas as pd
import copy

from scipy.special import softmax
    

class MHSBBO:
    """
   Simulation based Bayesian Optimization

    Args:

        - 
    """

    def __init__(self, co_problem, params):
        
        self.co_problem = co_problem
        self.af = params["af"]
        self.X = self.co_problem.X
        self.y = self.co_problem.y
        self.empty_model = params["model"]

        self.model = copy.deepcopy(self.empty_model)
        self.model.fit(self.X, self.y)

        self.cooling_schedule = params["cooling_schedule"]
        self.step = self.cooling_schedule[1] - self.cooling_schedule[0]
        self.burnin = params["burnin"]

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


    def update_all(self, i, temp, z_init, y_sample, value):

        # Update z
        for idx in range(self.co_problem.ncov):
            z_init, y_sample, value = self.metropolis_step(temp, value, y_sample, z_init)

        self.z_samples[i, ] = z_init

        return z_init, y_sample, value


    def iterate(self):

        z_init, y_sample, value = self.init_search()

        for i, temp in enumerate(self.cooling_schedule):
            if i%10 == 0:
                #print("Percentage completed:", 
                #np.round( 100*i/len(self.cooling_schedule), 2) )
                #print("Current state", z_init.reshape(5,-1))
                #print("Current energy", self.model.predict(self.co_problem.dummify(z_init.reshape(1,-1)) ))
                print(np.mean(y_sample))

            z_init, y_sample, value = self.update_all(i, temp, z_init, y_sample, value)
            

                
        z_star, quality = self.extract_solution()
        z_star_d = self.co_problem.dummify(z_star.reshape(1,-1))

        return z_star_d, quality
    
    def update(self, candidate, value):

        self.X = np.vstack([self.X, candidate])
        self.y = np.append(self.y, value)
        self.model = copy.deepcopy(self.empty_model)
        self.model.fit(self.X, self.y)



    def utility(self, y, z, flag='avg'):

        if flag == 'EI':
            result = np.zeros_like(y) 
            result[y > self.y.max()] = y[y > self.y.max()]  - self.y.max()
            return result + 0.0001

        elif flag == 'avg':
            return y
        
        elif flag == 'PI':
            result = np.zeros_like(y) 
            result[y >= self.y.max()] = 1.0
            return result + 0.0001

            
    def get_mode(self, samples):
        vals, counts = np.unique( samples, return_counts=True, axis=0 )
        index = np.argmax(counts)
        return vals[index]

    # CHECK!
    def extract_solution(self, modal=False):

        if modal:
            z_star = np.zeros_like(self.z_samples[0])
            burnin_end = int(self.burnin * len(self.cooling_schedule) )
            
            for j in range(z_star.shape[0]):
                z_star[j] = self.get_mode(self.z_samples[  burnin_end: , j ])

            # quality = self.co_problem.compute_obj(z_star)
            quality = self.model.predict(self.co_problem.dummify(z_star.reshape(1,-1)) )

        else:
            z_star = self.z_samples[-1]
            quality = self.model.predict(self.co_problem.dummify(z_star.reshape(1,-1)) )

        return z_star, quality





    
    