import numpy as np
import pandas as pd
from src.latin_square import LatinSquare
from src.sbbo import SBBO
from src.sbbo_mh import MHSBBO
from src.sbbo_gibbs import GibbsSBBO
from ngboost import NGBRegressor
from ngboost.distns import Exponential, Normal
from ngboost.scores import LogScore, CRPScore

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


opt_prob = LatinSquare(n=5)
lin_mod = LinearRegression()
dec_tree = DecisionTreeRegressor(criterion='friedman_mse', max_depth=5)

params = {}
# params["model"] = NGBRegressor()
# params["model"] = NGBRegressor(Dist=Normal, Base=lin_mod)
params["model"] = NGBRegressor(Dist=Normal, Base=lin_mod)
params["af"] = 'EI'
params["cooling_schedule"] = np.arange(1, 1000, 10)
params["burnin"] = 0.1
fname = "results/LS5_mNGBlin_oMH_afEI.csv"
method = MHSBBO(opt_prob, params)

method.co_problem.y.max()

sbbo = SBBO(opt_prob, method, fname)
sbbo.iterate(500)
