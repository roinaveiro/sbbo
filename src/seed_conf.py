import numpy as np

SEED_STR_LIST = ['CONTAMINATION']

def generate_random_seed_contamination():
    return _generate_random_seed('CONTAMINATION', n_init_point_seed=25)

def _generate_random_seed(seed_str, n_init_point_seed=25):
    assert seed_str in SEED_STR_LIST
    rng_state = np.random.RandomState(seed=sum([ord(ch) for ch in seed_str]))
    return rng_state.randint(0, 10000, (n_init_point_seed, ))

