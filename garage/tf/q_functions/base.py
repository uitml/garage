import tensorflow as tf

from garage.tf.core import Parameterized


class QFunction(Parameterized):
    def build_net(self, name):
        raise NotImplementedError

    def get_qval_sym(self, input_phs):
        raise NotImplementedError

    def log_diagnostics(self, paths):
        pass

    def get_trainable_vars(self, scope=None):
        scope = scope if scope else self.name
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    def get_global_vars(self, scope=None):
        scope = scope if scope else self.name
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    def get_regularizable_vars(self, scope=None):
        scope = scope if scope else self.name
        reg_vars = [
            var for var in self.get_trainable_vars(scope=scope)
            if 'W' in var.name and 'output' not in var.name
        ]
        return reg_vars

    def q_func(self,
               input_network,
               output_dim,
               hidden_sizes,
               name,
               hidden_nonlinearity=tf.nn.relu,
               hidden_w_init=tf.contrib.layers.xavier_initializer(),
               hidden_b_init=tf.zeros_initializer(),
               output_nonlinearity=None,
               output_w_init=tf.contrib.layers.xavier_initializer(),
               output_b_init=tf.zeros_initializer(),
               layer_normalization=False,
               dueling=False):
        """
        Q-Function.
        Useful for building q-function with another network as input, e.g. CNN.

        Args:
            input_network: Input tf.Tensor to the Q-Function.
            output_dim: Dimension of the network output.
            hidden_sizes: Output dimension of dense layer(s).
            name: variable scope of the Q-Function.
            hidden_nonlinearity: Activation function for
                        intermediate dense layer(s).
            hidden_w_init: Initializer function for the weight
                        of intermediate dense layer(s).
            hidden_b_init: Initializer function for the bias
                        of intermediate dense layer(s).
            output_nonlinearity: Activation function for
                        output dense layer.
            output_w_init: Initializer function for the weight
                        of output dense layer(s).
            output_b_init: Initializer function for the bias
                        of output dense layer(s).
            layer_normalization: Bool for using layer normalization or not.
            dueling: Boolean for using dueling network or not.

        Return:
            The output tf.Tensor of the Q-Function.
        """
        with tf.variable_scope(name):
            with tf.variable_scope("action_value"):
                l_hid = input_network
                for idx, hidden_size in enumerate(hidden_sizes):
                    l_hid = tf.layers.dense(
                        inputs=l_hid,
                        units=hidden_size,
                        activation=hidden_nonlinearity,
                        kernel_initializer=hidden_w_init,
                        bias_initializer=hidden_b_init,
                        name="action_value")
                    if layer_normalization:
                        l_hid = tf.contrib.layers.layer_norm(l_hid)
                action_out = tf.layers.dense(
                    inputs=l_hid,
                    units=output_dim,
                    activation=output_nonlinearity,
                    kernel_initializer=output_w_init,
                    bias_initializer=output_b_init,
                    name="output_action_value")

            if dueling:
                with tf.variable_scope("state_value"):
                    l_hid = input_network
                    for idx, hidden_size in enumerate(hidden_sizes):
                        l_hid = tf.layers.dense(
                            inputs=l_hid,
                            units=hidden_size,
                            activation=hidden_nonlinearity,
                            kernel_initializer=hidden_w_init,
                            bias_initializer=hidden_b_init,
                            name="state_value")
                        if layer_normalization:
                            l_hid = tf.contrib.layers.layer_norm(l_hid)
                    state_out = tf.layers.dense(
                        inputs=l_hid,
                        units=output_dim,
                        activation=output_nonlinearity,
                        kernel_initializer=output_w_init,
                        bias_initializer=output_b_init,
                        name="output_state_value")
                action_out_mean = tf.reduce_mean(action_out, 1)
                # calculate the advantage of performing certain action
                # over other action in a particular state
                action_out_advantage = action_out - tf.expand_dims(
                    action_out_mean, 1)
                q_out = state_out + action_out_advantage
            else:
                q_out = action_out

        return q_out
