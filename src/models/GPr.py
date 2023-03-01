import numpy as np
import pandas as pd
import copy

import gpflow
from gpflow.mean_functions import Constant
from gpflow.utilities import positive, print_summary
from gpflow.utilities.ops import broadcasting_elementwise
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
# from rdkit.Chem import AllChem, Descriptors, MolFromSmiles
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from ngboost.distns import Normal



class Tanimoto(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        # We constrain the value of the kernel variance to be positive when it's being optimised
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):
        """
        Compute the Tanimoto kernel matrix σ² * ((<x, y>) / (||x||^2 + ||y||^2 - <x, y>))
        :param X: N x D array
        :param X2: M x D array. If None, compute the N x N kernel matrix for X.
        :return: The kernel matrix of dimension N x M
        """
        if X2 is None:
            X2 = X

        Xs = tf.reduce_sum(tf.square(X), axis=-1)  # Squared L2-norm of X
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)  # Squared L2-norm of X2
        outer_product = tf.tensordot(X, X2, [[-1], [-1]])  # outer product of the matrices X and X2

        # Analogue of denominator in Tanimoto formula

        denominator = -outer_product + broadcasting_elementwise(tf.add, Xs, X2s)

        return self.variance * outer_product/denominator

    def K_diag(self, X):
        """
        Compute the diagonal of the N x N kernel matrix of X
        :param X: N x D array
        :return: N x 1 array
        """
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))
    
def transform_data(X_train, y_train):
    """
    Apply feature scaling to the data. Return the standardised train and
    test sets together with the scaler object for the target values.
    :param X_train: input train data
    :param y_train: train labels
    :return: X_train_scaled, y_train_scaled, y_scaler
    """

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)


    return X_train_scaled, y_train_scaled, y_scaler
  

class GPr:
    """
   GP regression with Jaccard Kernel

    Args:

        - 
    """

    def __init__(self, kernel='Tanimoto', noise=0.00001):
        
        if kernel == 'Tanimoto':
            self.kernel = Tanimoto()
        else:
            pass

        self.noise=noise

    def fit(self, X, y):

        y = y.reshape(-1, 1)
        _, y_scaled, self.y_scaler = transform_data(X, y)
        X = X.astype(np.float64)

    
        self.m = gpflow.models.GPR(data=(X, y_scaled), mean_function=Constant(np.mean(y)), 
                              kernel=self.kernel, noise_variance=self.noise)
        
        
        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.m.training_loss, self.m.trainable_variables)


    def pred_dist(self, X):

        X = X.astype(np.float64)
        y_pred, y_var = self.m.predict_f(X)
        y_pred = self.y_scaler.inverse_transform(y_pred)

        y_lstd = np.log( np.sqrt(y_var.numpy().squeeze()) )
        y_pred = y_pred.squeeze()

        params = np.array([y_pred, y_lstd])

        return Normal(params)

    def predict(self, X):

        X = X.astype(np.float64)
        y_pred, _ = self.m.predict_f(X)
        y_pred = self.y_scaler.inverse_transform(y_pred)
        return y_pred.squeeze()
