"""Deterministic MLP."""
import tensorflow as tf

from garage.core import Serializable
from garage.misc.overrides import overrides
from garage.tf.core import Parameterized
from garage.tf.core.mlp import mlp
from garage.tf.misc import tensor_utils


class DeterministicMLP(Parameterized, Serializable):
    """Deterministic Multi-Layer Perceptron."""

    def __init__(self,
                 output_size,
                 input_size=None,
                 name="DeterministicMLP",
                 hidden_sizes=(32, 32),
                 input_dtype=tf.float32,
                 input_layer=None,
                 hidden_nonlinearity=tf.nn.tanh,
                 output_nonlinearity=None,
                 **kwargs):
        """Initializer.

        :param output_size size of output layer.
        :param input_size size of input layer.
        :param name name of MLP.
        :param hidden_sizes tuple of sizes for fully-connected hidden layers.
        :param input_dtype data type of the input layer.
        :param hidden_nonlinearity nonlinearity used for each hidden layer.
        :param output_nonlinearity nonlinearity for the output layer.
        :return:
        """
        assert output_size > 0
        assert input_size > 0
        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())

        self._name = name
        self._name_scope = tf.name_scope(self._name)

        # Network parameters
        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._output_nonlinearity = output_nonlinearity
        self._output_size = output_size

        # Build default graph
        with self._name_scope:
            # inputs
            if input_layer is not None:
                self._input_sym = input_layer
            elif input_size is not None:
                self._input_sym = tf.placeholder(
                    input_dtype, shape=(None, input_size), name="input")
            else:
                raise ValueError("Please specify input_size or input_layer")

            with tf.name_scope("default", values=[self._input_sym]):
                self._output_sym = self._build_graph(**kwargs)

            # compiled functions
            with tf.variable_scope("operation"):
                self._f_output = tensor_utils.compile_function(
                    inputs=[self._input_sym],
                    outputs=[self._output_sym],
                )

    @property
    def input_sym(self):
        """Get input variable."""
        return self._input_sym

    @property
    def output_sym(self):
        """Get output variable."""
        return self._output_sym

    def get_output(self, inputs):
        """Run MLP graph and get output."""
        return self._f_output(inputs)

    def _build_graph(self, **kwargs):
        """Build the computation graph."""
        output_var = mlp(
            input_var=self._input_sym,
            output_dim=self._output_size,
            hidden_sizes=self._hidden_sizes,
            name="MLP",
            hidden_nonlinearity=self._hidden_nonlinearity,
            output_nonlinearity=self._output_nonlinearity,
            **kwargs)
        return output_var

    @overrides
    def get_params_internal(self, **tags):
        """Get intenal params."""
        if tags.get("trainable"):
            params = [v for v in tf.trainable_variables(scope=self._name)]
        else:
            params = [v for v in tf.global_variables(scope=self._name)]
        return params
