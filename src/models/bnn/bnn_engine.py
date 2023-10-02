import ml_collections
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import sklearn.metrics

from . import  training



class BNN_engine(tf.keras.models.Model):

    def __init__(self, layer_size: int, n_layers: int, kld_beta: float, output_dim: int):
        super(BNN_engine, self).__init__()

        self.kld_beta = kld_beta

        self.bnn_layers = []
        cnn = False
        simple = True
        super_simple = False

        if cnn: 
            self.bnn_layers.append(tf.keras.layers.Reshape([25, 1]))
            self.bnn_layers.append( tfp.layers.Convolution1DReparameterization(8, 
                kernel_size=5, padding='SAME', activation=tf.nn.relu) )
            self.bnn_layers.append( tf.keras.layers.Flatten() )
            self.bnn_layers.append( tfp.layers.DenseReparameterization(5, activation='relu') )

        elif simple:
            self.bnn_layers.append(tfp.layers.DenseReparameterization(10, activation='relu'))
            self.bnn_layers.append(tfp.layers.DenseReparameterization(5, activation='relu'))
        
        elif super_simple:
            self.bnn_layers.append(tfp.layers.DenseReparameterization(4, activation='relu'))


        else:
            self.bnn_layers.append(tfp.layers.DenseReparameterization(60, activation='relu',
                activity_regularizer = tf.keras.regularizers.L2(0.01)))
            self.bnn_layers.append(tfp.layers.DenseReparameterization(30, activation='relu',
                activity_regularizer = tf.keras.regularizers.L2(0.01)))
            self.bnn_layers.append(tfp.layers.DenseReparameterization(10, activation='relu',
                activity_regularizer = tf.keras.regularizers.L2(0.01)))
            

        #for _ in range(n_layers):
        #    self.bnn_layers.append(tfp.layers.DenseReparameterization(layer_size, activation='relu'))
            # self.bnn_layers.append(tf.keras.layers.BatchNormalization())

        self.out_layer = tf.keras.layers.Dense(output_dim)
        self.act = tf.identity

    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        for i in range(len(self.bnn_layers)):
            x = self.bnn_layers[i](x, training=training)
        x_out = self.out_layer(x)
        output = self.act(x_out)
        return output

    def train(self, x_train, y_train, x_val, y_val, hp, verbose=True):
        optimizer = tf.keras.optimizers.legacy.Adam(hp.lr)
        main_metric, metric_fn = ('R^2', sklearn.metrics.r2_score)

        early_stop = training.EarlyStopping(self, patience=hp.patience)
        stop_metric = f'val_{main_metric}'
        pbar = tqdm(range(hp.epochs), disable=not verbose)
        stats = []


        w_train, w_val = None, None

        # bnn uses kld regularized loss
        n = len(y_train)

        def loss_fn(y_true, y_pred, sample_weight=None):
            fn = tf.keras.losses.MeanSquaredError()
            scaled_kl = tf.math.reduce_mean(self.losses) / n
            loss = fn(y_true, y_pred, sample_weight=sample_weight) + self.kld_beta * scaled_kl
            return loss

        # start training
        for _ in pbar:
            # mini-batch training
            training.train_step(self, x_train, y_train, optimizer, loss_fn, hp.batch_size, sample_weight=w_train)

            # evalulate all sets
            result = {}
            for inputs, target, weight, prefix in [
                (x_train, y_train, w_train, 'train'),
                (x_val, y_val, w_val, 'val'),
            ]:
                output = self(inputs)
                result[f'{prefix}_{main_metric}'] = metric_fn(target, output, sample_weight=weight)
                result[f'{prefix}_loss'] = loss_fn(target, output, sample_weight=weight).numpy()
            stats.append(result)

            pbar.set_postfix(stats[-1])
            if early_stop.check_criteria(stats[-1][stop_metric]):
                break

        early_stop.restore_best()
        best_step = early_stop.best_step
        print(f'Early stopped at {best_step} with {stop_metric}={stats[best_step][stop_metric]:.3f}')

    
    def predict(self, x, n_pred=1000):

        # predict values and the standard deviation (MC)
        predictions = []
        for _ in range(n_pred):
            predictions.append(self(x).numpy())
        # predictions = np.concatenate(predictions, axis = 1)
        predictions = np.stack(predictions, axis=1)

        # y_mu = np.mean(predictions, axis=0).reshape(y.shape)
        # y_std = np.std(predictions, axis=0).reshape(y.shape)
        
        return predictions.reshape(-1, n_pred)



    @classmethod
    def from_hparams(cls, hp) -> 'BNN':
        return cls(layer_size=hp.layer_size,
                   n_layers=hp.n_layers,
                   kld_beta=hp.kld_beta,
                   output_dim=hp.output_dim)


def default_hp(output_dim=1):
    hp = ml_collections.ConfigDict()
    hp.layer_size = 5
    hp.n_layers = 2
    hp.output_dim = output_dim
    hp.patience = 100
    hp.lr = 1e-3
    hp.epochs = 2000
    hp.batch_size = 128
    hp.kld_beta = 0.1
    return hp
