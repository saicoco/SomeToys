# -*- coding: utf-8 -*-
# author = sai

from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import gzip, cPickle
import numpy as np

# load mnist
def OneHot(data):
    shape = len(data)
    labels = np.zeros((shape, 10))
    for i in xrange(shape):
        labels[i, data[i]] = 1
    return labels

def load_mnist(one_hot=False, train=True):
    with gzip.open('../mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)
        train_x, train_y = train_set
        valid_x, valid_y = valid_set
        test_x, test_y = test_set
    if train:
        data = train_x
        labels = train_y
    else:
        data = test_x
        labels = test_y
    if one_hot:
        labels = OneHot(labels)
    return data, labels
# config
# parameter
lr = 0.001
training_iters = 100000
batch_size = 128
dis = 10
epoches = 20

# network parameters
n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

# tf graph input
x = tf.placeholder('float', [None, n_steps, n_input])
y = tf.placeholder('float', [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}

biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input]) # reshape to n_steps*batch_size
    x = tf.split(0, n_steps, x)

    # Define a lstm ccell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outpus, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outpus[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# define loss and optimazer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimazer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

# evaluate model
correct_pre = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

# initializing the variables
init = tf.initialize_all_variables()
# launch the graph
with tf.Session() as sess:
    sess.run(init)
    epoch = 1
    train_x, train_y = load_mnist(one_hot=True)
    while epoch < epoches:
        step = 1
        acc = 0.0
        loss = 0.0
        while step * batch_size < train_x.shape[0]:
            batch_x = train_x[(step-1)*batch_size:step*batch_size]
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            batch_y = train_y[(step - 1) * batch_size:step * batch_size]
            sess.run(optimazer, feed_dict={x:batch_x, y:batch_y})

            acc += sess.run(accuracy, feed_dict={x:batch_x, y:batch_y})
            loss += sess.run(cost, feed_dict={x:batch_x, y:batch_y})
            step += 1
        print("Iter " + str(epoch) + ", Minibatch Loss= " + \
              "{:.6f}".format(loss/step) + ", Training Accuracy= " + \
              "{:.5f}".format(acc/step))
        epoch += 1
    print('optimazation finished!')

    test_len = 128
    test_data, test_labels = load_mnist(one_hot=True, train=False)
    test_data = test_data.reshape((test_data.shape[0], 28, 28))
    print('Testing accuracy:', sess.run(accuracy, feed_dict={x:test_data, y:test_labels}))




