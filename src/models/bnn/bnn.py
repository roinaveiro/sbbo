import numpy as np
from .bnn_engine import BNN_engine, default_hp
from src.config import get_multi_sampler

from sklearn.model_selection import train_test_split


class BNN:

    def __init__(self):

        self.hp = default_hp()

    def fit(self, X, y):

        X = X.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(X, y, 
            test_size=0.2, random_state=83)


        self.hp.batch_size = int(X_train.shape[0] / 10) + 1 
        self.hp.kld_beta = 2.0
        self.model = BNN_engine.from_hparams(self.hp)
        self.model.train(X_train, y_train, X_test, y_test, 
            self.hp, verbose=True)

    
    def sample_pred(self, X, n_pred=10):
        return self.model.predict(X, n_pred)

    def predict(self, X):
        pred_samples = self.sample_pred(X)
        return np.mean(pred_samples, axis=1)

    def pred_dist(self, X):
        X = X.astype(float)
        pred_samples = self.sample_pred(X)
        samplers = get_multi_sampler(pred_samples)

        return samplers

         

