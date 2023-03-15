import argparse
import os

import numpy as np
import pandas as pd

from src.problems.latin_square import LatinSquare
from src.problems.contamination import Contamination
from src.problems.bqp import BQP

from src.sbbo import SBBO

from src.sbbo_mh import MHSBBO
from src.sbbo_gibbs import GibbsSBBO
from src.sbbo_rs import RS
from src.sbbo_sa import SA

from ngboost import NGBRegressor
from ngboost.distns import Exponential, Normal, LogNormal
from ngboost.scores import LogScore, CRPScore

# Learners
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LassoCV
from src.models.GPr import GPr
from src.models.bocs.LinReg import LinReg


from src.seed_conf import generate_random_seed_contamination


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True, type=str, default="CON")
    parser.add_argument("--learner", required=True, type=str, default="dec")
    parser.add_argument("--acqfun", required=True, type=str, default="EI")
    parser.add_argument("--search", required=True, type=str, default="MH")
    parser.add_argument("--niters", type=int, default=500)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--nexp", type=int, default=0)
    parser.add_argument("--seed_conf", type=int, default=None)

    return parser.parse_args()
  

if __name__ == "__main__":

    args = parse_args()

    random_seed_config_ = args.seed_conf
    if random_seed_config_ is not None:
        assert 0 <= int(random_seed_config_) <= 24
        #random_seed_config_ -= 1


    params = {}
    params["cooling_schedule"] = np.arange(1, 1000, 10)
    params["burnin"] = 0.8
    params["modal"]  = False

    if args.problem == "CON":
        if random_seed_config_ is not None:
            random_seed = generate_random_seed_contamination()
            seed_ = random_seed[random_seed_config_]
            print("Seed", seed_)
            # init_seed_ = sorted(random_seed_pair_[case_seed_])[int(random_seed_config_ % 5)]
            opt_prob = Contamination(lamda=0.0001, n=5, random_seed=seed_)
        else:
            opt_prob = Contamination(lamda=0.0001, n=5, random_seed=None)
    elif args.problem == "BQP":
        if random_seed_config_ is not None:
            random_seed = generate_random_seed_contamination()
            seed_ = random_seed[random_seed_config_]
            print("Seed", seed_)
            # init_seed_ = sorted(random_seed_pair_[case_seed_])[int(random_seed_config_ % 5)]
            opt_prob = BQP(n=5, random_seed=seed_)
        else:
            opt_prob = BQP(n=5, random_seed=None)

    elif args.problem == "LS5":
        opt_prob = LatinSquare(n=5)

    if args.learner == "NGBdec":
        learner = DecisionTreeRegressor(criterion='friedman_mse', max_depth=5)
        params["model"] = NGBRegressor(Dist=Normal, Base=learner)
    elif args.learner == "NGBlin":
        learner = LinearRegression()
        params["model"] = NGBRegressor(Dist=Normal, Base=learner)
    elif args.learner == "NGBlinCV":
        learner = LassoCV(cv=5)
        params["model"] = NGBRegressor(Dist=Normal, Base=learner)
    elif args.learner == "GPr":
        learner = GPr()
        params["model"] = learner
    elif args.learner == "BOCS":
        learner = LinReg(nVars=opt_prob.ncov, order=2)
        params["model"] = learner
    elif args.learner == "BOCS_NS":
        learner = LinReg(nVars=opt_prob.ncov, order=2)
        params["model"] = learner

    if args.acqfun == "EI":
        params["af"] = "EI"
    elif args.acqfun == "AVG":
        params["af"] = "AVG"
    elif args.acqfun == "PI":
        params["af"] = "PI"

    if args.search == "MH":
        search = MHSBBO(opt_prob, params)
    elif args.search == "Gibbs":
        search = GibbsSBBO(opt_prob, params)
    elif args.search == "RS":
        search = RS(opt_prob, params)
    elif args.search == "SA":
        search = SA(opt_prob, params)


    root_folder = f"results/{args.problem}/"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    folder = f"results/{args.problem}/{args.search}/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    if args.search == "SA":
        fname = f"{folder}exp{args.nexp}_{args.problem}_o{args.search}_seed{args.seed_conf}.csv"

    elif args.search == "RS":
        fname = f"{folder}exp{args.nexp}_{args.problem}_o{args.search}_seed{args.seed_conf}.csv"

    else:
        fname = f"{folder}exp{args.nexp}_{args.problem}_m{args.learner}_o{args.search}_af{args.acqfun}_seed{args.seed_conf}.csv"
    
    print("Write to...", fname)

    sbbo = SBBO(opt_prob, search, fname, args.epsilon)

    sbbo.iterate(args.niters)
