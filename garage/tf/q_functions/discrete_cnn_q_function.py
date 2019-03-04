"""Discrete MLP QFunction."""
import tensorflow as tf

from garage.misc.overrides import overrides
from garage.tf.models.discrete_mlp_model import DiscreteMLPModel
from garage.tf.models.discrete_cnn_q_model import DiscreteCNNQModel
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
                 hidden_sizes,
                 name="DiscreteCNNQFunction",
                 padding="SAME",
                 max_pooling=False,
                 pool_shape=(2, 2),
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.contrib.layers.xavier_initializer,
                 hidden_b_init=tf.zeros_initializer,
                 output_nonlinearity=None,
                 output_w_init=tf.contrib.layers.xavier_initializer,
                 output_b_init=tf.zeros_initializer,
                 dueling=False,
                 layer_normalization=False):
        self.name = name
        self.action_dim = env_spec.action_space.n
        self._hidden_sizes = hidden_sizes

        self._filter_dims = filter_dims
        self._num_filters = num_filters
        self._strides = strides
        self._padding = padding

        out_model = DiscreteMLPModel(
            output_dim=self.action_dim,
            name=name+'_mlp',
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization)

        self.model = DiscreteCNNQModel(
            name=name+'_cnn',
            filter_dims=filter_dims,
            num_filters=num_filters,
            strides=strides,
            padding=padding,
            out_model=out_model)

        obs_dim = env_spec.observation_space.shape
        obs_ph = tf.placeholder(tf.float32, (None, ) + obs_dim, name="obs")
        self.model.build(obs_ph)

    def build_net(self, input_var, name):
        out_model = DiscreteMLPModel(
            output_dim=self.action_dim,
            name=name+'_mlp',
            hidden_sizes=self._hidden_sizes)

        model = DiscreteCNNQModel(
            name=name+'_cnn',
            filter_dims=self._filter_dims,
            num_filters=self._num_filters,
            strides=self._strides,
            padding=self._padding,
            out_model=out_model)

        return model.build(input_var)

    @overrides
    def get_qval_sym(self, state_input, name):
        """
        Symbolic graph for q-network.

        Args:
            state_input: The state input tf.Tensor to the network.
            name: Network variable scope.
        """
        return self.model.build(state_input, name=name)