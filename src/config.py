import numpy as np
import pandas as pd


RNA_CHAIN_LEN = 30
########################################################################################################################

CONTAMINATION_N_STAGES = 25

def generate_contamination_dynamics(random_seed=None):
    n_stages = CONTAMINATION_N_STAGES
    n_simulations = 100

    init_alpha = 1.0
    init_beta = 30.0
    contam_alpha = 1.0
    contam_beta = 17.0 / 3.0
    restore_alpha = 1.0
    restore_beta = 3.0 / 7.0
    init_Z = np.random.RandomState(random_seed).beta(init_alpha, init_beta, size=(n_simulations,))
    lambdas = np.random.RandomState(random_seed).beta(contam_alpha, contam_beta, size=(n_stages, n_simulations))
    gammas = np.random.RandomState(random_seed).beta(restore_alpha, restore_beta, size=(n_stages, n_simulations))

    return init_Z, lambdas, gammas


class Sampler(object):

    def __init__(self, samples) -> None:
        self.samples = samples

    def sample(self, n_samples):

        assert 1 <= n_samples <= self.samples.shape[0]
        samples_perm = self.samples[np.random.permutation(np.arange(self.samples.shape[0]))]

        return samples_perm[:n_samples]
    

def get_multi_sampler(samples):

    if samples.shape[0] == 1:
        return Sampler(samples[0])
    else:
        samplers = []
        for i in range(samples.shape[0]):
            samplers.append(Sampler(samples[i]))

        return samplers
