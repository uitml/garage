import gym
import tensorflow as tf

from garage.envs import normalize
from garage.tf.envs import TfEnv
from garage.tf.q_functions.discrete_cnn_q_function import DiscreteCNNQFunction
from garage.tf.q_functions.discrete_mlp_q_function import DiscreteMLPQFunction
from tests.fixtures import TfGraphTestCase


class TestQFunction(TfGraphTestCase):
    def test_mlp_q_function_output(self):

        env = TfEnv(normalize(gym.make('CartPole-v0')))

        q_function = DiscreteMLPQFunction(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=None,
            layer_norm=False)

        # Construct the input placeholder
        obs_dim = env.spec.observation_space.shape
        obs_ph = tf.placeholder(tf.float32, (None, ) + obs_dim, name="obs")

        # Build the net
        q_net = q_function.build_net(name="q_net", input=obs_ph)

        self.sess.run(tf.global_variables_initializer())

        result = self.sess.run(q_net, feed_dict={obs_ph: [env.reset()]})

        assert result.shape == (1, env.spec.action_space.n)

    def test_cnn_q_function_output(self):

        env = TfEnv(normalize(gym.make('Breakout-v0')))

        q_function = DiscreteCNNQFunction(
            env_spec=env.spec, filter_dims=(8, 4, 3), num_filters=(16, 32, 32))

        # Construct the input placeholder
        obs_dim = env.spec.observation_space.shape
        obs_ph = tf.placeholder(tf.float32, (None, ) + obs_dim, name="obs")

        # Build the net
        q_net = q_function.build_net(name="q_net", input=obs_ph)

        self.sess.run(tf.global_variables_initializer())

        result = self.sess.run(q_net, feed_dict={obs_ph: [env.reset()]})

        assert result.shape == (1, env.spec.action_space.n)
