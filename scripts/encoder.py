import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import backend

from pathlib import Path

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import logging

#logging.basicConfig(level=logging.DEBUG)

import pandas as pd
import numpy as np

import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

class CenteredGaussianNoise(tf.keras.layers.Layer):
    """Apply additive non zero-centered Gaussian noise.

    As a regularization layer, it is only activate at training time."""

    def __init__(self, center, stddev, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev
        self.center = center

    def call(self, inputs, training=None):

        def noised():
            return inputs + backend.random_normal(
                shape=tf.shape(inputs),
                mean=self.center,
                stddev=self.stddev,
            )
        return backend.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'center': self.center,
                  'stddev': self.stddev}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

class OneCycleLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, iterations, max_rate, start_rate=None,
                 last_iterations=None, last_rate=None):
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0

    def _interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (self.iteration - iter1)
                / (iter2 - iter1) + rate1)

    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                     self.max_rate, self.start_rate)
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                     self.start_rate, self.last_rate)
        self.iteration += 1
        backend.set_value(self.model.optimizer.learning_rate, rate)


def onecycle_lr(eta0, n_epochs, n_drop, model):
    '''One cycle learning rate function with a linear ramp-up, linear, ramp-down, and drop in n_drop epoch

    At the end n_drop epochs, the learning rate drops linearly by a factor of 10 (relative to eta0) every 5 steps

    Function from HOML
    '''

    def onecycle_lr_fn(epoch, lr):
        if epoch <= n_epochs / 2:
            lr0 = backend.get_value(model.optimizer.learning_rate)
            inc = 18 * eta0 / n_epochs
            return (lr0 + inc)
        elif epoch > n_epochs / 2 and epoch <= n_epochs - n_drop:
            lr0 = backend.get_value(model.optimizer.learning_rate)
            dec = (9 * eta0) / (n_epochs / 2 - n_drop)
            return (lr0 - dec)
        else:
            lr0 = backend.get_value(model.optimizer.learning_rate)
            sharp_dec = 0.9 * eta0 / 5
            return (lr0 - sharp_dec)

    return onecycle_lr_fn

class OneCycleMomentumCallback(tf.keras.callbacks.Callback):
    def __init__(self, n_epochs, n_drop):
        self.n_epochs = n_epochs
        self.n_drop = n_drop
    def on_epoch_end(self, epoch, logs=None):
        if epoch <= self.n_epochs/2:
            m0 = backend.get_value(self.model.optimizer.beta_1)
            dec = 0.2 / self.n_epochs
            backend.set_value(self.model.optimizer.beta_1, (m0 - dec))
        elif epoch > self.n_epochs/2 and epoch <= self.n_epochs - self.n_drop:
            m0 = backend.get_value(self.model.optimizer.beta_1)
            inc = 0.1 / (self.n_epochs/2 - self.n_drop)
            backend.set_value(self.model.optimizer.beta_1, (m0 + inc))
        else:
            backend.set_value(self.model.optimizer.beta_1, 0.95)


def exponential_decay(lr0, s):
    """Exponential decay function from HOML"""
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn

def decayed_learning_rate(lr0, s):
    def decayed_learning_rate_fn(epoch):
        return lr0 / (1 + epoch / s)
    return decayed_learning_rate_fn


class DenseTranspose(tf.keras.layers.Layer):
    def __init__(self, dense, activation=None, kernel_initializer=None, kernel_regularizer=None, **kwargs):
        self.dense = dense
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        super().__init__(**kwargs)

    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias", initializer="zeros",
                                      shape=[self.dense.input_shape[-1]])
        super().build(batch_input_shape)

    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)


def monotonic(x):
    sorted = x.sort_values(ascending=False).index
    return np.array_equal(sorted.values, ['A', 'C', 'D', 'B']) or np.array_equal(sorted.values, ['B', 'D', 'C', 'A'])


def assess_monotonicity(data, metadata):
    """Returns percentage of genes that follow titration monotonicity -
    higher is better"""
    data['sample'] = metadata.loc[data.index, 'seqc_sample']
    grouped = data.groupby('sample').mean()
    data.drop('sample', inplace=True, axis=1)

    return grouped.apply(monotonic, axis=0).mean()


def assess_sem(data, metadata):
    """mean SEM across technical replicates <10% - returns percentage of genes (higher is better)"""
    data['sample'] = metadata.loc[data.index, 'seqc_sample']
    grouped_mean = data.groupby('sample').mean()
    grouped_sem = data.groupby('sample').sem()
    sem_mean = grouped_sem.div(grouped_mean).mean()
    data.drop('sample', inplace=True, axis=1)

    return np.mean(sem_mean < 0.1)


def expected_behavior(x, z=1.43):
    k1 = 3 * z / (3 * z + 1)
    k2 = z / (z + 3)

    t1 = k1 * (x['A'] / x['B']) + (1 - k1)
    t2 = k2 * (x['A'] / x['B']) + (1 - k2)
    return np.log(t1) - np.log(t2)


def assess_behavior(data, metadata):
    """% of genes where deviation less than expected behavior
    (SEQC/MAQC-III Consortium, 2014, Nat Biotech;Shippy, 2006, Nat Biotech)"""
    data['sample'] = metadata.loc[data.index, 'seqc_sample']
    grouped_mean = data.groupby('sample').mean()
    # remove genes where mean for one group is zero
    grouped_zeros = grouped_mean.apply(lambda x: all(x > 0))
    # add minimum value so that latent space assessment has no zeros
    grouped_meanz = grouped_mean.loc[:, grouped_zeros]+np.min(grouped_mean.values)

    logcd = grouped_meanz.apply(lambda x: np.log(x['C'] / x['D'])).reindex()
    logab = grouped_meanz.apply(expected_behavior, axis=0).reindex()
    data.drop('sample', inplace=True, axis=1)

    return np.mean([logab[i] * 0.9 <= logcd[i] <= logab[i] * 1.1 for i in logab.index])

def build_autoencoder(train, valid, config):
    """Build autoencoder with parameters in the configdict"""
    # calculate layers
    layered = pd.Series(reversed(range(config['nlayers'])))
    layers_scaling = ((config['nfeatures'] - config['nlatent']) / config['nlayers'])
    if (layers_scaling < 0):
        raise Exception('Number of input features smaller than number of latent features')
    enc_layers = layered * layers_scaling + config['nlatent']

    # gaussian noise layer setup with mean 0.5
    # centered_gaussian_noise_layer = tf.keras.layers.Lambda(lambda x: x + tf.random.normal(tf.shape(x),
    #                                                                                    mean=0.5,
    #                                                                                    stddev=config[
    #                                                                                        'noise_std'],
    #                                                                                    dtype=x.dtype))

    enc = Sequential(name='encoder')

    enc.add(Dropout(config['dropout_rate'], input_shape=(train.shape[1],)))
    if config['noise']:
        # enc.add(centered_gaussian_noise_layer)
        enc.add(CenteredGaussianNoise(center=0.5, stddev=config['noise_std']))

    enc_list = []
    for i in range(config['nlayers']):
        enc_list.append(Dense(units=enc_layers[i], activation="elu",
                              kernel_initializer="he_normal",
                              kernel_regularizer=tf.keras.regularizers.l2(config['l2reg'])))
        enc.add(enc_list[i])
        if (config['batchnorm']):
            enc.add(BatchNormalization())

    dec = Sequential(name='decoder')

    # DenseTranspose links weights to weights of encoder (see pp 577 HOML)
    for i in range(1, config['nlayers']):
        dec.add(DenseTranspose(enc_list[-i], activation="elu",
                                  kernel_initializer="he_normal",
                                  kernel_regularizer=tf.keras.regularizers.l2(config['l2reg'])))
        if (config['batchnorm']):
            dec.add(BatchNormalization())
    dec.add(DenseTranspose(enc_list[-config['nlayers']], activation="sigmoid"))

    return tf.keras.models.Sequential([enc, dec])

def run_autoencoder(datapath, datarun, dataprefix, config, metafile, outputdir, neptuneapi):
    x_train = pd.read_csv(datapath / (datarun + '-train' + dataprefix + ".csv"),
                          index_col=0).astype(
        'float32')
    x_valid = pd.read_csv(datapath / (datarun + '-valid' + dataprefix + ".csv"),
                          index_col=0).astype(
        'float32')

    metadata = pd.read_csv(datapath / metafile, index_col=0, sep='\t')

    feature_order = x_train.var(axis=0).sort_values(ascending=False).index

    features = feature_order[0:config['nfeatures']]
    train = x_train.loc[:, features]
    valid = x_valid.loc[:, features]


    model = build_autoencoder(train, valid, config)

    model.compile(loss=tf.keras.losses.mean_squared_error,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']))

    nrun = neptune.init(project='mariah.hoffman/ModelChecker-SEQC-Autoencoder002', #mode="debug")
                        api_token=neptuneapi)

    nrun["config"] = config
    nrun["directory/data"] = datapath
    nrun["directory/output"] = outputdir

    neptune_cbk = NeptuneCallback(run=nrun, base_namespace='metrics')

    # assume power scheduler
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        decayed_learning_rate(lr0=config['lr'], s=config['s']))
    callbacks = [lr_scheduler, neptune_cbk]


    history = model.fit(train, train,
                        batch_size=config['batchsize'],
                        epochs=1000,
                        verbose=0,
                        callbacks=callbacks,
                        validation_data=(valid, valid))

    # mean loss since model is sum of individual
    train_score = model.evaluate(train, train, verbose=0) / config['nfeatures']
    valid_score = model.evaluate(valid, valid, verbose=0) / config['nfeatures']

    latent = tf.keras.Model(inputs=[model.layers[0].input], outputs=[model.layers[0].output])
    val = latent.predict(valid)

    # Calculate the SEQC-specific metrics
    latent_val = pd.DataFrame(val, index=valid.index)
    monotonicity = 1 - assess_monotonicity(latent_val, metadata)
    sems = 1 - assess_sem(latent_val, metadata)
    try:
        expected_behavior = 1 - assess_behavior(latent_val, metadata)
    except:

        expected_behavior = None

    metricsdict = {
        'valid loss': valid_score,
        'train loss': train_score,
        'monotonicity': monotonicity,
        'SEM metric': sems,
        'expected behavior': expected_behavior
    }

    nrun["metrics/validmetrics"] = metricsdict
    nrun.stop()

    model.save(Path(outputdir) / "model")

    return history, model, features

def prefix_to_scaling(prefix):
    if ('sm' in prefix) or ('ss' in prefix):
        return 'Feature-wise'
    elif ('gm' in prefix) or ('gs' in prefix):
        return 'Global'


