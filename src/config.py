import numpy as np
import pandas as pd


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