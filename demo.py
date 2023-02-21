import numpy as np
import pandas as pd
from src.latin_square import LatinSquare
from src.sbbo_mh import SBBO
from ngboost import NGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


opt_prob = LatinSquare()
ngb = NGBRegressor()
BO = SBBO(opt_prob, ngb, np.arange(1, 1000))

a,b,c = BO.iterate()

print("Done")

