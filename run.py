import argparse
import os

import numpy as np
import pandas as pd

from src.latin_square import LatinSquare
from src.contamination import Contamination

from src.sbbo import SBBO

from src.sbbo_mh import MHSBBO
from src.sbbo_gibbs import GibbsSBBO
from src.sbbo_rs import RS

from ngboost import NGBRegressor
from ngboost.distns import Exponential, Normal
from ngboost.scores import LogScore, CRPScore

# Learners
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LassoCV
from src.models.GPr import GPr


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True, type=str, default="CON")
    parser.add_argument("--learner", required=True, type=str, default="dec")
    parser.add_argument("--acqfun", required=True, type=str, default="EI")
    parser.add_argument("--search", required=True, type=str, default="MH")
    parser.add_argument("--niters", type=int, default=500)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--nexp", type=int, default=0)

    return parser.parse_args()
  

if __name__ == "__main__":

    args = parse_args()

    params = {}
    params["cooling_schedule"] = np.arange(1, 1000, 10)
    params["burnin"] = 0.6

    if args.problem == "CON":
        opt_prob = Contamination(lamda=0.0001, n=5)
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


    root_folder = f"results/{args.problem}/"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    folder = f"results/{args.problem}/{args.search}/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    fname = f"{folder}exp{args.nexp}_{args.problem}_m{args.learner}_o{args.search}_af{args.acqfun}.csv"
    
    print("Write to...", fname)

    sbbo = SBBO(opt_prob, search, fname, args.epsilon)

    sbbo.iterate(args.niters)
