# This file defines the UFRL networks

import tensorflow as tf
from tensorflow.contrib import layers


class Net_UFRL(object):
    def __init__(self, v, dims_net, activation, reg=None):

        self.v = v
        self.dims_net = dims_net
        self.num_layers = len(self.dims_net)
        self.activation = activation
        self.reg = reg

        if activation in ['tanh', 'sigmoid']:
            self.initializer = layers.xavier_initializer()
        if activation == 'relu':
            self.initializer = layers.xavier_initializer()

        self.weights, self.netpara = self.init_weights()

    def init_weights(self):
        all_weights = dict()
        with tf.variable_scope("lxnet"):
            for i in range(1, self.num_layers):
                all_weights['lx' + str(self.v) + '_w' + str(i)] = tf.get_variable("lx" + str(self.v) + "_w" + str(i),
                                                                                  shape=[self.dims_net[i - 1],
                                                                                         self.dims_net[i]],
                                                                                  initializer=self.initializer,
                                                                                  regularizer=self.reg)
                all_weights['lx' + str(self.v) + '_b' + str(i)] = tf.Variable(
                    tf.zeros([self.dims_net[i]], dtype=tf.float32))

            lxnet = tf.trainable_variables()
        return all_weights, lxnet

    def learning_view_specific(self, h, weights):
        layer = tf.add(tf.matmul(h, weights['lx' + str(self.v) + '_w1']), weights['lx' + str(self.v) + '_b1'])
        if self.activation == 'sigmoid':
            layer = tf.nn.sigmoid(layer)
        if self.activation == 'tanh':
            layer = tf.nn.tanh(layer)
        if self.activation == 'relu':
            layer = tf.nn.relu(layer)
        for i in range(2, self.num_layers):
            layer = tf.add(tf.matmul(layer, weights['lx' + str(self.v) + '_w' + str(i)]),
                           weights['lx' + str(self.v) + '_b' + str(i)])

            if self.activation == 'sigmoid':
                layer = tf.nn.sigmoid(layer)
            if self.activation == 'tanh':
                layer = tf.nn.tanh(layer)
            if self.activation == 'relu':
                layer = tf.nn.relu(layer)
        return layer

    def loss_mse_reconstruction(self, h, z_half):
        g = self.learning_view_specific(h, self.weights)
        loss = tf.losses.mean_squared_error(z_half, g)
        return loss

    def get_x(self, h):
        return self.learning_view_specific(h, self.weights)

