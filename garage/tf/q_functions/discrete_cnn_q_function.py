"""Discrete MLP QFunction."""
import tensorflow as tf

from garage.misc.overrides import overrides
from garage.tf.core.cnn import cnn
from garage.tf.core.cnn import cnn_with_max_pooling
from garage.tf.core.mlp import mlp
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
                 strides,
                 name="DiscreteCNNQFunction",
                 padding="SAME",
                 max_pooling=False,
                 pool_shape=(2, 2),
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.contrib.layers.xavier_initializer(),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.contrib.layers.xavier_initializer(),
                 output_b_init=tf.zeros_initializer(),
                 dueling=False,
                 layer_norm=False):
        super().__init__()

        self.name = name
        self._env_spec = env_spec
        self._action_dim = env_spec.action_space.n
        self._filter_dims = filter_dims
        self._num_filters = num_filters
        self._strides = strides
        self._padding = padding
        self._max_pooling = max_pooling
        self._pool_shape = pool_shape
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_norm = layer_norm
        self._dueling = dueling

        obs_dim = self._env_spec.observation_space.shape
        self.obs_ph = tf.placeholder(tf.float32, (None, ) + obs_dim, name="obs")
        self.q_val = self.build_net(name, self.obs_ph)

    @overrides
    def build_net(self, name, input_var):
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
        with tf.variable_scope(name):
            if self._max_pooling:
                input_var = cnn_with_max_pooling(
                    input_var=input_var,
                    output_dim=512,
                    filter_dims=self._filter_dims,
                    hidden_nonlinearity=self._hidden_nonlinearity,
                    output_nonlinearity=self._hidden_nonlinearity,
                    num_filters=self._num_filters,
                    stride=self._strides,
                    padding=self._padding,
                    max_pooling=self._max_pooling,
                    pool_shape=self._pool_shape,
                    name="cnn")
            else:
                input_var = cnn(
                    input_var=input_var,
                    output_dim=512,
                    filter_dims=self._filter_dims,
                    hidden_nonlinearity=self._hidden_nonlinearity,
                    output_nonlinearity=self._hidden_nonlinearity,
                    num_filters=self._num_filters,
                    strides=self._strides,
                    padding=self._padding,
                    name="cnn")

            dueling_hidden_sizes = [256]

            action_out = mlp(
                input_var=input_var,
                output_dim=self._action_dim,
                hidden_sizes=dueling_hidden_sizes,
                name="action_value",
                hidden_nonlinearity=self._hidden_nonlinearity,
                hidden_w_init=self._hidden_w_init,
                hidden_b_init=self._hidden_b_init,
                output_nonlinearity=self._output_nonlinearity,
                output_w_init=self._output_w_init,
                output_b_init=self._output_b_init,
                layer_normalization=self._layer_norm)

            if self._dueling:
                state_out = mlp(
                    input_var=input_var,
                    output_dim=1,
                    hidden_sizes=dueling_hidden_sizes,
                    name="state_value",
                    hidden_nonlinearity=self._hidden_nonlinearity,
                    hidden_w_init=self._hidden_w_init,
                    hidden_b_init=self._hidden_b_init,
                    output_nonlinearity=self._output_nonlinearity,
                    output_w_init=self._output_w_init,
                    output_b_init=self._output_b_init,
                    layer_normalization=self._layer_norm)

                action_out_mean = tf.reduce_mean(action_out, 1)
                # calculate the advantage of performing certain action
                # over other action in a particular state
                action_out_advantage = action_out - tf.expand_dims(
                    action_out_mean, 1)
                q_func_out = state_out + action_out_advantage
            else:
                q_func_out = action_out

            return q_func_out

    def get_qval_sym(self, name, input_var):
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
        with tf.variable_scope(name):
            if self._max_pooling:
                input_var = cnn_with_max_pooling(
                    input_var=input_var,
                    output_dim=512,
                    filter_dims=self._filter_dims,
                    hidden_nonlinearity=self._hidden_nonlinearity,
                    output_nonlinearity=self._hidden_nonlinearity,
                    num_filters=self._num_filters,
                    stride=self._strides,
                    padding=self._padding,
                    max_pooling=self._max_pooling,
                    pool_shape=self._pool_shape,
                    name="cnn",
                    reuse=True)
            else:
                input_var = cnn(
                    input_var=input_var,
                    output_dim=512,
                    filter_dims=self._filter_dims,
                    hidden_nonlinearity=self._hidden_nonlinearity,
                    output_nonlinearity=self._hidden_nonlinearity,
                    num_filters=self._num_filters,
                    strides=self._strides,
                    padding=self._padding,
                    name="cnn",
                    reuse=True)

            dueling_hidden_sizes = [256]

            action_out = mlp(
                input_var=input_var,
                output_dim=self._action_dim,
                hidden_sizes=dueling_hidden_sizes,
                name="action_value",
                hidden_nonlinearity=self._hidden_nonlinearity,
                hidden_w_init=self._hidden_w_init,
                hidden_b_init=self._hidden_b_init,
                output_nonlinearity=self._output_nonlinearity,
                output_w_init=self._output_w_init,
                output_b_init=self._output_b_init,
                layer_normalization=self._layer_norm,
                reuse=True)

            if self._dueling:
                state_out = mlp(
                    input_var=input_var,
                    output_dim=1,
                    hidden_sizes=dueling_hidden_sizes,
                    name="state_value",
                    hidden_nonlinearity=self._hidden_nonlinearity,
                    hidden_w_init=self._hidden_w_init,
                    hidden_b_init=self._hidden_b_init,
                    output_nonlinearity=self._output_nonlinearity,
                    output_w_init=self._output_w_init,
                    output_b_init=self._output_b_init,
                    layer_normalization=self._layer_norm,
                    reuse=True)

                action_out_mean = tf.reduce_mean(action_out, 1)
                # calculate the advantage of performing certain action
                # over other action in a particular state
                action_out_advantage = action_out - tf.expand_dims(
                    action_out_mean, 1)
                q_func_out = state_out + action_out_advantage
            else:
                q_func_out = action_out

            return q_func_out