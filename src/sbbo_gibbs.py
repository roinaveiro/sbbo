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
        self.step = self.cooling_schedule[1] - self.cooling_schedule[0]
        self.burnin = burnin

        self.z_samples = np.zeros( [len(self.cooling_schedule),
                            self.co_problem.ncov] )
 


    def init_search(self):

        z_init = self.co_problem.generate_candidate()
        z_init_d = self.co_problem.dummify(z_init)
        preds = self.model.pred_dist(z_init_d)
        y_sample = preds[0].sample(self.cooling_schedule[0])
        value = self.utility(y_sample, z_init)

        return z_init, y_sample, value


    def update_probs_metropolis(self, value_old, y_sample_old, z):

        z_d = self.co_problem.dummify(z)
        #--#
        preds = self.model.pred_dist(z_d)
        y_sample_new = preds[0].sample(len(y_sample_old))
        value_new = self.utility(y_sample_new, z)

        condition = np.random.uniform(size=len(value_new)) < value_new / value_old
        y_sample_old[condition] = y_sample_new[condition] 
        value_old[condition] = value_new[condition]

        return y_sample_old, value_old

      
    def update_z(self, z, idx, y_sample):

        Z_candidates = self.co_problem.generate_candidates_idx(z,idx)
        Z_candidates_d = self.co_problem.dummify(Z_candidates)
        preds = self.model.pred_dist(Z_candidates_d)

        energies = np.zeros(Z_candidates.shape[0])

        for i in range(len(energies)):
            energies[i] = np.sum(preds[i].dist.logpdf(y_sample)) # Utilities??

        p = softmax(energies)
        candidate_idx = np.random.choice( np.arange(len(Z_candidates)), p=p )
        return Z_candidates[candidate_idx]



    def update_all(self, i, temp, z_init, y_sample, value):

        # Update z
        for idx in range(self.co_problem.ncov):
            z_init = self.update_z(z_init, idx, y_sample)
            self.z_samples[i, idx] = z_init[idx]
            z_init = z_init.reshape(1,-1)

        y_sample, value = self.update_probs_metropolis(value,
                                                         y_sample, z_init)

        return z_init, y_sample, value


    def iterate(self):

        z_init, y_sample, value = self.init_search()

        for i, temp in enumerate(self.cooling_schedule):
            if i%1 == 0:
                # print("Percentage completed:", 
                # np.round( 100*i/len(self.cooling_schedule), 2) )
                # print("Current state", z_init.reshape(5,-1))
                # print("Current energy", self.model.predict(self.co_problem.dummify(z_init.reshape(1,-1)) ))
                print(np.mean(y_sample))

            z_init, y_sample, value = self.update_all(i, temp, z_init, y_sample, value)
            y_sample = np.append(y_sample, np.random.choice(y_sample, self.step))
            value = self.utility(y_sample, z_init)
            
        z_star, quality = self.extract_solution()
        
        return z_star, self.z_samples, quality



    def utility(self, y, z, flag='avg'):

        if flag == 'EI':

            result = np.zeros_like(y) + 0.0001
            result[y > self.y.max()] = y[y > self.y.max()]  - self.y.max()
            return result

        elif flag == 'avg':
            return y
        
        elif flag == 'PI':
            result = np.zeros_like(y) 
            result[y > self.y.max()] = 1.0
            return result + 0.0001

            


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





    
    