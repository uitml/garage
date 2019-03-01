"""LSTM network in TensorFlow."""

import tensorflow as tf


def lstm(input_var,
         output_dim,
         hidden_dim,
         hidden_w_init=tf.contrib.layers.xavier_initializer(),
         hidden_b_init=tf.zeros_initializer(),
         output_nonlinearity=None,
         output_w_init=tf.contrib.layers.xavier_initializer(),
         output_b_init=tf.zeros_initializer()
         name='lstm'):
    with tf.variable_scope(name):
         