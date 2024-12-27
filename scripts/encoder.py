import tensorflow as tf

#logging.basicConfig(level=logging.DEBUG)

import pandas as pd
import numpy as np

from typeguard import typechecked
from typing import Optional, Union, List


## From tensorflow_addons

from packaging.version import Version

# Find KerasTensor.
if Version(tf.__version__).release >= Version("2.16").release:
    # Determine if loading keras 2 or 3.
    if (
        hasattr(tf.keras, "version")
        and Version(tf.keras.version()).release >= Version("3.0").release
    ):
        from keras import KerasTensor
    else:
        from tf_keras.src.engine.keras_tensor import KerasTensor
elif Version(tf.__version__).release >= Version("2.13").release:
    from keras.src.engine.keras_tensor import KerasTensor
elif Version(tf.__version__).release >= Version("2.5").release:
    from keras.engine.keras_tensor import KerasTensor
else:
    from tensorflow.python.keras.engine.keras_tensor import KerasTensor

## From tensorflow_addons
Number = Union[
    float,
    int,
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]
TensorLike = Union[
    List[Union[Number, list]],
    tuple,
    Number,
    np.ndarray,
    tf.Tensor,
    tf.SparseTensor,
    tf.Variable,
    KerasTensor,
]
FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]
AcceptableDTypes = Union[tf.DType, np.dtype, type, int, str, None]

## From tensorflow_addons
def pairwise_distance(feature: TensorLike, squared: bool = False):
    """Computes the pairwise distance matrix with numerical stability.

    output[i, j] = || feature[i, :] - feature[j, :] ||_2

    Args:
      feature: 2-D Tensor of size `[number of data, feature dimension]`.
      squared: Boolean, whether or not to square the pairwise distances.

    Returns:
      pairwise_distances: 2-D Tensor of size `[number of data, number of data]`.
    """
    pairwise_distances_squared = tf.math.add(
        tf.math.reduce_sum(tf.math.square(feature), axis=[1], keepdims=True),
        tf.math.reduce_sum(
            tf.math.square(tf.transpose(feature)), axis=[0], keepdims=True
        ),
    ) - 2.0 * tf.matmul(feature, tf.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = tf.math.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = tf.math.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = tf.math.sqrt(
            pairwise_distances_squared
            + tf.cast(error_mask, dtype=tf.dtypes.float32) * 1e-16
        )

    # Undo conditionally adding 1e-16.
    pairwise_distances = tf.math.multiply(
        pairwise_distances,
        tf.cast(tf.math.logical_not(error_mask), dtype=tf.dtypes.float32),
    )

    num_data = tf.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = tf.ones_like(pairwise_distances) - tf.linalg.diag(
        tf.ones([num_data])
    )
    pairwise_distances = tf.math.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances

## From tensorflow_addons
def is_tensor_or_variable(x):
    return tf.is_tensor(x) or isinstance(x, tf.Variable)

class LossFunctionWrapper(tf.keras.losses.Loss):
    """Wraps a loss function in the `Loss` class."""

    def __init__(
        self, fn, reduction=tf.keras.losses.Reduction.AUTO, name=None, **kwargs
    ):
        """Initializes `LossFunctionWrapper` class.

        Args:
          fn: The loss function to wrap, with signature `fn(y_true, y_pred,
            **kwargs)`.
          reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used with
            `tf.distribute.Strategy`, outside of built-in training loops such as
            `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial](
              https://www.tensorflow.org/tutorials/distribute/custom_training)
            for more details.
          name: (Optional) name for the loss.
          **kwargs: The keyword arguments that are passed on to `fn`.
        """
        super().__init__(reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        """Invokes the `LossFunctionWrapper` instance.

        Args:
          y_true: Ground truth values.
          y_pred: The predicted values.

        Returns:
          Loss values per sample.
        """
        return self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        config = {}
        for k, v in iter(self._fn_kwargs.items()):
            config[k] = tf.keras.backend.eval(v) if is_tensor_or_variable(v) else v
        base_config = super().get_config()
        return {**base_config, **config}


## From tensorflow_addons
def lifted_struct_loss(
    labels: TensorLike, embeddings: TensorLike, margin: FloatTensorLike = 1.0
) -> tf.Tensor:
    """Computes the lifted structured loss.

    Args:
      labels: 1-D tf.int32 `Tensor` with shape `[batch_size]` of
        multiclass integer labels.
      embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
        not be l2 normalized.
      margin: Float, margin term in the loss definition.

    Returns:
      lifted_loss: float scalar with dtype of embeddings.
    """
    convert_to_float32 = (
        embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
    )
    precise_embeddings = (
        tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
    )

    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = tf.shape(labels)
    labels = tf.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    pairwise_distances = pairwise_distance(precise_embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = tf.math.equal(labels, tf.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = tf.math.logical_not(adjacency)

    batch_size = tf.size(labels)

    diff = margin - pairwise_distances
    mask = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
    # Safe maximum: Temporarily shift negative distances
    #   above zero before taking max.
    #     this is to take the max only among negatives.
    row_minimums = tf.math.reduce_min(diff, 1, keepdims=True)
    row_negative_maximums = (
        tf.math.reduce_max(
            tf.math.multiply(diff - row_minimums, mask), 1, keepdims=True
        )
        + row_minimums
    )

    # Compute the loss.
    # Keep track of matrix of maximums where M_ij = max(m_i, m_j)
    #   where m_i is the max of alpha - negative D_i's.
    # This matches the Caffe loss layer implementation at:
    #   https://github.com/rksltnl/Caffe-Deep-Metric-Learning-CVPR16/blob/0efd7544a9846f58df923c8b992198ba5c355454/src/caffe/layers/lifted_struct_similarity_softmax_layer.cpp

    max_elements = tf.math.maximum(
        row_negative_maximums, tf.transpose(row_negative_maximums)
    )
    diff_tiled = tf.tile(diff, [batch_size, 1])
    mask_tiled = tf.tile(mask, [batch_size, 1])
    max_elements_vect = tf.reshape(tf.transpose(max_elements), [-1, 1])

    loss_exp_left = tf.reshape(
        tf.math.reduce_sum(
            tf.math.multiply(tf.math.exp(diff_tiled - max_elements_vect), mask_tiled),
            1,
            keepdims=True,
        ),
        [batch_size, batch_size],
    )

    loss_mat = max_elements + tf.math.log(loss_exp_left + tf.transpose(loss_exp_left))
    # Add the positive distance.
    loss_mat += pairwise_distances

    mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
        tf.ones([batch_size])
    )

    # *0.5 for upper triangular, and another *0.5 for 1/2 factor for loss^2.
    num_positives = tf.math.reduce_sum(mask_positives) / 2.0

    lifted_loss = tf.math.truediv(
        0.25
        * tf.math.reduce_sum(
            tf.math.square(
                tf.math.maximum(tf.math.multiply(loss_mat, mask_positives), 0.0)
            )
        ),
        num_positives,
    )

    if convert_to_float32:
        return tf.cast(lifted_loss, embeddings.dtype)
    else:
        return lifted_loss
    
class LiftedStructLoss(LossFunctionWrapper):
    """Computes the lifted structured loss.

    The loss encourages the positive distances (between a pair of embeddings
    with the same labels) to be smaller than any negative distances (between
    a pair of embeddings with different labels) in the mini-batch in a way
    that is differentiable with respect to the embedding vectors.
    See: https://arxiv.org/abs/1511.06452.

    Args:
      margin: Float, margin term in the loss definition.
      name: Optional name for the op.
    """

    @typechecked
    def __init__(
        self, margin: FloatTensorLike = 1.0, name: Optional[str] = None, **kwargs
    ):
        super().__init__(
            lifted_struct_loss,
            name=name,
            reduction=tf.keras.losses.Reduction.NONE,
            margin=margin,
        )

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
            return inputs + tf.keras.backend.random_normal(
                shape=tf.shape(inputs),
                mean=self.center,
                stddev=self.stddev,
            )
        return tf.keras.backend.in_train_phase(noised, inputs, training=training)

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
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, rate)

def onecycle_lr(eta0, n_epochs, n_drop, model):
    '''One cycle learning rate function with a linear ramp-up, linear, ramp-down, and drop in n_drop epoch

    At the end n_drop epochs, the learning rate drops linearly by a factor of 10 (relative to eta0) every 5 steps

    Function from HOML
    '''

    def onecycle_lr_fn(epoch, lr):
        if epoch <= n_epochs / 2:
            lr0 = tf.keras.backend.get_value(model.optimizer.learning_rate)
            inc = 18 * eta0 / n_epochs
            return (lr0 + inc)
        elif epoch > n_epochs / 2 and epoch <= n_epochs - n_drop:
            lr0 = tf.keras.backend.get_value(model.optimizer.learning_rate)
            dec = (9 * eta0) / (n_epochs / 2 - n_drop)
            return (lr0 - dec)
        else:
            lr0 = tf.keras.backend.get_value(model.optimizer.learning_rate)
            sharp_dec = 0.9 * eta0 / 5
            return (lr0 - sharp_dec)

    return onecycle_lr_fn

class OneCycleMomentumCallback(tf.keras.callbacks.Callback):
    def __init__(self, n_epochs, n_drop):
        self.n_epochs = n_epochs
        self.n_drop = n_drop
    def on_epoch_end(self, epoch, logs=None):
        if epoch <= self.n_epochs/2:
            m0 = tf.keras.backend.get_value(self.model.optimizer.beta_1)
            dec = 0.2 / self.n_epochs
            tf.keras.backend.set_value(self.model.optimizer.beta_1, (m0 - dec))
        elif epoch > self.n_epochs/2 and epoch <= self.n_epochs - self.n_drop:
            m0 = tf.keras.backend.get_value(self.model.optimizer.beta_1)
            inc = 0.1 / (self.n_epochs/2 - self.n_drop)
            tf.keras.backend.set_value(self.model.optimizer.beta_1, (m0 + inc))
        else:
            tf.keras.backend.set_value(self.model.optimizer.beta_1, 0.95)

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

def build_autoencoder(train, valid, config):
    """Build autoencoder with parameters in the configdict"""
    # calculate layers
    layered = pd.Series(reversed(range(config['nlayers'])))
    layers_scaling = ((config['nfeatures'] - config['nlatent']) / config['nlayers'])
    if (layers_scaling < 0):
        raise Exception('Number of input features smaller than number of latent features')
    enc_layers = layered * layers_scaling + config['nlatent']

    enc = tf.keras.models.Sequential(name='encoder')

    enc.add(tf.keras.layers.Dropout(config['dropout_rate'], input_shape=(train.shape[1],)))
    if config['noise'] == 'True':
        enc.add(CenteredGaussianNoise(center=0, stddev=config['noise_std']))

    enc_list = []
    for i in range(config['nlayers']):
        enc_list.append(tf.keras.layers.Dense(units=enc_layers[i], activation="elu",
                              kernel_initializer="he_normal",
                              kernel_regularizer=tf.keras.regularizers.l2(config['l2reg'])))
        enc.add(enc_list[i])
        if (config['batchnorm'] == "True"):
            enc.add(tf.keras.layers.BatchNormalization())

    dec = tf.keras.models.Sequential(name='decoder')

    # DenseTranspose links weights to weights of encoder (see pp 577 HOML)
    for i in range(1, config['nlayers']):
        dec.add(DenseTranspose(enc_list[-i], activation="elu",
                                  kernel_initializer="he_normal",
                                  kernel_regularizer=tf.keras.regularizers.l2(config['l2reg'])))
        if (config['batchnorm'] == "True"):
            dec.add(tf.keras.layers.BatchNormalization())
    dec.add(DenseTranspose(enc_list[-config['nlayers']], activation="sigmoid"))

    return tf.keras.models.Sequential([enc, dec])

def build_contrastiverep(train, valid, config):
    """Build contrastive representation learning model with parameters in the configdict"""
    # calculate layers
    layered = pd.Series(reversed(range(config['nlayers'])))
    layers_scaling = ((config['nfeatures'] - config['nlatent']) / config['nlayers'])
    if (layers_scaling < 0):
        raise Exception('Number of input features smaller than number of latent features')
    enc_layers = layered * layers_scaling + config['nlatent']

    enc = tf.keras.models.Sequential(name='encoder')

    enc.add(tf.keras.layers.Dropout(config['dropout_rate'], input_shape=(train.shape[1],)))
    if config['noise'] == 'True':
        enc.add(CenteredGaussianNoise(center=0, stddev=config['noise_std']))

    enc_list = []
    for i in range(config['nlayers']):
        enc_list.append(tf.keras.layers.Dense(units=enc_layers[i], activation="elu",
                              kernel_initializer="he_normal",
                              kernel_regularizer=tf.keras.regularizers.l2(config['l2reg'])))
        enc.add(enc_list[i])
        if (config['batchnorm'] == "True"):
            enc.add(tf.keras.layers.BatchNormalization())

    return tf.keras.models.Sequential([enc])