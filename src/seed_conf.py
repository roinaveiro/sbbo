import numpy as np

SEED_STR_LIST = ['CONTAMINATION']

def generate_random_seed_pair_contamination():
    return _generate_random_seed_pair('CONTAMINATION', n_test_case_seed=5, n_init_point_seed=5)


def _generate_random_seed_pair(seed_str, n_test_case_seed=5, n_init_point_seed=5):
    assert seed_str in SEED_STR_LIST
    rng_state = np.random.RandomState(seed=sum([ord(ch) for ch in seed_str]))
    result = {}
    for _ in range(n_test_case_seed):
        result[rng_state.randint(0, 10000)] = list(rng_state.randint(0, 10000, (n_init_point_seed, )))
    return result