"""Discrete MLP QFunction."""
import tensorflow as tf

from garage.misc.overrides import overrides
from garage.tf.core.cnn import cnn
from garage.tf.core.cnn import cnn_with_max_pooling
from garage.tf.q_functions import QFunction


class DiscreteCNNQFunction(QFunction):
    """
    Q function based on CNN for discrete action space.

    This class implements a Q value network to predict Q based on the
    input state and action. It uses an CNN to fit the function of Q(s, a).

    Args:
        env_spec: environment specification
        filter_dims: Dimension of the filters.
        num_filters: Number of filters.
        stride: The stride of the sliding window.
        name: Variable scope of the cnn.
        padding: The type of padding algorithm to use, from "SAME", "VALID".
        max_pooling: Boolean for using max pooling layer or not.
        pool_shape: Dimension of the pooling layer(s).
        hidden_nonlinearity: Activation function for
                    intermediate dense layer(s).
        output_nonlinearity: Activation function for
                    output dense layer.
    """

    def __init__(self,
                 env_spec,
                 filter_dims,
                 num_filters,
                 name="DiscreteCNNQFunction",
                 stride=1,
                 padding="SAME",
                 max_pooling=False,
                 pool_shape=(2, 2),
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=None):
        super().__init__()

        self.name = name
        self._env_spec = env_spec
        self._action_dim = env_spec.action_space.n
        self._filter_dims = filter_dims
        self._hidden_nonlinearity = hidden_nonlinearity
        self._output_nonlinearity = output_nonlinearity
        self._num_filters = num_filters
        self._stride = stride
        self._padding = padding
        self._max_pooling = max_pooling
        self._pool_shape = pool_shape

        self.obs_ph = self._build_ph()
        self.q_val = self.build_net(name, self.obs_ph)

    def _build_ph(self):
        obs_dim = self._env_spec.observation_space.shape
        obs_ph = tf.placeholder(tf.float32, (None, ) + obs_dim, name="obs")
        return obs_ph

    @overrides
    def build_net(self, name, input_var, dueling=False, layer_norm=False):
        """
        Build the q-network.

        Args:
            name: scope of the network.
            input_var: input Tensor of the network.
            dueling: use dueling network or not.
            layer_norm: Boolean for layer normalization.

        Return:
            The tf.Tensor of Discrete CNNQFunction.
        """
        if self._max_pooling:
            network = cnn_with_max_pooling(
                input_var=input_var,
                output_dim=512,
                filter_dims=self._filter_dims,
                hidden_nonlinearity=self._hidden_nonlinearity,
                output_nonlinearity=self._hidden_nonlinearity,
                num_filters=self._num_filters,
                stride=self._stride,
                padding=self._padding,
                max_pooling=self._max_pooling,
                pool_shape=self._pool_shape,
                name=name)
        else:
            network = cnn(
                input_var=input_var,
                output_dim=512,
                filter_dims=self._filter_dims,
                hidden_nonlinearity=self._hidden_nonlinearity,
                output_nonlinearity=self._hidden_nonlinearity,
                num_filters=self._num_filters,
                stride=self._stride,
                padding=self._padding,
                name=name)

        return self.q_func(
            input_network=network,
            output_dim=self._action_dim,
            hidden_sizes=[256],
            name=name,
            hidden_nonlinearity=self._hidden_nonlinearity,
            output_nonlinearity=self._output_nonlinearity,
            layer_normalization=layer_norm,
            dueling=dueling)
