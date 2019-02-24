"""Discrete MLP QFunction."""
import tensorflow as tf

from garage.misc.overrides import overrides
from garage.tf.core.mlp import mlp
from garage.tf.q_functions import QFunction


class DiscreteMLPQFunction(QFunction):
    """
    Discrete MLP Function.

    This class implements a Q-value network. It predicts Q-value based on the
    input state and action. It uses an MLP to fit the function Q(s, a).

    Args:
        env_spec: environment specification
        hidden_sizes: A list of numbers of hidden units
            for all hidden layers.
        hidden_nonlinearity: An activation shared by all fc layers.
        output_nonlinearity: An activation used by the output layer.
        layer_norm: A bool to indicate whether to perform
            layer normalization or not.
    """

    def __init__(self,
                 env_spec,
                 name="discrete_mlp_q_function",
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.contrib.layers.xavier_initializer(),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.contrib.layers.xavier_initializer(),
                 output_b_init=tf.zeros_initializer(),
                 dueling=False,
                 layer_norm=False):

        self.name = name
        self._env_spec = env_spec
        self._action_dim = env_spec.action_space.n
        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_norm = layer_norm
        self._dueling = dueling

        self.obs_ph = self._build_ph()
        self.q_val = self.build_net(name, self.obs_ph)

    def _build_ph(self):
        obs_dim = self._env_spec.observation_space.shape
        return tf.placeholder(tf.float32, (None, ) + obs_dim, name="obs")

    @overrides
    def build_net(self, name, input_var):
        """
        Build the q-network.

        Args:
            name: scope of the network.
            input_var: input Tensor of the network.

        Return:
            The tf.Tensor of Discrete DiscreteMLPQFunction.
        """
        if isinstance(self._hidden_sizes, int):
            dueling_hidden_sizes = [self._hidden_sizes]
        else:
            dueling_hidden_sizes = [self._hidden_sizes[-1]]
        with tf.variable_scope(name, reuse=False):
            if not isinstance(self._hidden_sizes, int):
                input_var = mlp(
                    input_var=input_var,
                    output_dim=self._hidden_sizes[-2],
                    hidden_sizes=self._hidden_sizes[:-2],
                    name="mlp",
                    hidden_nonlinearity=self._hidden_nonlinearity,
                    hidden_w_init=self._hidden_w_init,
                    hidden_b_init=self._hidden_b_init,
                    output_nonlinearity=self._hidden_nonlinearity,
                    output_w_init=self._hidden_w_init,
                    output_b_init=self._hidden_b_init,
                    layer_normalization=self._layer_norm)

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

    @overrides
    def get_qval_sym(self, name, input_var):
        """
        Symbolic graph for the q-value network.

        Args:
            input_var: input Tensor of the network.

        Return:
            The tf.Tensor of Discrete DiscreteMLPQFunction.
        """
        if isinstance(self._hidden_sizes, int):
            dueling_hidden_sizes = [self._hidden_sizes]
        else:
            dueling_hidden_sizes = [self._hidden_sizes[-1]]
        with tf.variable_scope(name, reuse=True):
            if not isinstance(self._hidden_sizes, int):
                input_var = mlp(
                    input_var=input_var,
                    output_dim=self._hidden_sizes[-2],
                    hidden_sizes=self._hidden_sizes[:-2],
                    name="mlp",
                    hidden_nonlinearity=self._hidden_nonlinearity,
                    hidden_w_init=self._hidden_w_init,
                    hidden_b_init=self._hidden_b_init,
                    output_nonlinearity=self._hidden_nonlinearity,
                    output_w_init=self._hidden_w_init,
                    output_b_init=self._hidden_b_init,
                    layer_normalization=self._layer_norm)

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
