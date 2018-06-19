# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.rnn import BasicRNNCell
from tensorflow.examples.tutorials.mnist import input_data


import numpy as np
import os
from datetime import datetime

# <codecell>

import horovod.tensorflow as hvd

# <codecell>

n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10

# <codecell>

hvd.init()

# <codecell>

config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())

# <codecell>

tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

basic_cell = BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

logits = fully_connected(states, n_outputs, activation_fn=None)

with tf.name_scope('loss'):
    x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits, name='x_entropy')
    loss = tf.reduce_mean(x_entropy, name ='loss')

learning_rate = 0.01
with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = hvd.DistributedOptimizer(optimizer)
    hooks = [hvd.BroadcastGlobalVariablesHook(0)]

    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# <codecell>

n_epochs = 10
batch_size = 50
mnist = input_data.read_data_sets("data")

# <codecell>

X_test = mnist.test.images.reshape(-1, n_steps, n_inputs)
y_test = mnist.test.labels

# <codecell>

with tf.train.MonitoredTrainingSession(config=config, hooks=hooks) as mon_sess:
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape(-1, n_steps, n_inputs)
            _, acc_train = mon_sess.run([training_op, accuracy], feed_dict={X:X_batch, y:y_batch})
        print(epoch, "Train accuracy ", acc_train)


# <codecell>


