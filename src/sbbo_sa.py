import numpy as np
from scipy.special import softmax

class SA:
    """
   Simulation based Bayesian Optimization - RS

    Args:

        - 
    """

    def __init__(self, co_problem, params):

        self.co_problem = co_problem
        self.X = self.co_problem.X
        self.y = self.co_problem.y

        self.z_old = self.X[np.argmax(self.y)]
        self.old_energy = np.max(self.y)
        self.iter = 0
    
    def generate_candidate(self):
        
        z_old_aux = self.z_old.reshape(1,-1)
        idx = np.random.choice(np.arange(z_old_aux.shape[1]))

        Z_candidates = self.co_problem.generate_candidates_idx(z_old_aux, idx)
        
        # Remove z_old and choose one at random
        new_Z_candidates = Z_candidates[~np.all(Z_candidates == z_old_aux, axis=1)]
        z_new = new_Z_candidates[np.random.randint(new_Z_candidates.shape[0]), :]

        z_candidate_d = self.co_problem.dummify(z_new)
        new_energy = self.co_problem.compute_obj(z_candidate_d)

        prob = np.exp( (new_energy - self.old_energy) / self.s() )

        if np.random.uniform() < prob:
            print("Accept!")
            return z_new, new_energy
        
        else:
            
            return self.z_old, self.old_energy

    def iterate(self):

        self.z_old, self.old_energy = self.generate_candidate()
        self.iter += 1
        return self.z_old, self.old_energy
    
    def update(self, candidate, value):
        self.X = np.vstack([self.X, candidate])
        self.y = np.append(self.y, value)

    def s(self, den=25, l=5):
        return np.exp(-l*self.iter/den) # Watch out!
    



if __name__ == "__main__":

    pass

