import numpy as np
import pandas as pd

from src.config import CONTAMINATION_N_STAGES, generate_contamination_dynamics


def _contamination(x, cost, init_Z, lambdas, gammas, U, epsilon):
    assert x.size == CONTAMINATION_N_STAGES

    rho = 1.0
    n_simulations = 100


    Z = np.zeros((x.size, n_simulations))
    Z[0] = lambdas[0] * (1.0 - x[0]) * (1.0 - init_Z) + (1.0 - gammas[0] * x[0]) * init_Z
    for i in range(1, CONTAMINATION_N_STAGES):
        Z[i] = lambdas[i] * (1.0 - x[i]) * (1.0 - Z[i - 1]) + (1.0 - gammas[i] * x[i]) * Z[i - 1]

    below_threshold = Z < U
    constraints = np.mean(below_threshold, axis=1) - (1.0 - epsilon)

    return np.sum(x * cost - rho * constraints)


class Contamination(object):
    """
    Contamination Control Problem with the simplest graph
    """
    def __init__(self, lamda, n=10, random_seed_pair=(129534, 128593)):
        self.lamda = lamda
        self.n_init = n
        self.ncov = CONTAMINATION_N_STAGES

        self.scaler = 5.0
       
        # In all evaluation, the same sampled values are used.
        self.init_Z, self.lambdas, self.gammas = generate_contamination_dynamics(random_seed_pair[0])
        self.generate_init_data()

    def evaluate(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.array([self.compute_obj(x[i]) for i in range(x.shape[0])])

    def compute_obj(self, x, scale=True):
        if x.ndim == 2:
            x = x.squeeze()

        assert x.ndim == 1
        evaluation = _contamination(x, cost=np.ones_like(x), init_Z=self.init_Z, 
                                    lambdas=self.lambdas, gammas=self.gammas,
                                      U=0.1, epsilon=0.05)
        evaluation += self.lamda * float(np.sum(x))

        if scale:
            return -evaluation + self.ncov
        else:
            return evaluation
    
    def generate_init_data(self, n=100):
        self.X = np.random.choice([0,1],
                                   size=CONTAMINATION_N_STAGES*self.n_init).reshape(self.n_init,-1)
        self.y = self.evaluate(self.X)


    def generate_candidate(self):
        return np.random.choice([0,1], size=CONTAMINATION_N_STAGES).reshape(1,-1)

    def generate_candidates_idx(self, z, idx):
        result = np.repeat(z, 2, axis=0)
        result[:, idx] = np.arange(2)
        return result
    
    def desdummify(self, x):
        return x

    def dummify(self, x):
        return x



    


    
    