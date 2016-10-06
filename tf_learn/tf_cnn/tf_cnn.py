# -*- coding: utf-8 -*-
# author = sai
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

def conv2d(x, w, b, strides=1):
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def convnet(x, weights, biases, dropout):
    x = tf.reshape(x, [-1, 28, 28, 1])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2)

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# config
# parameter
lr = 0.001
batch_size = 128
dis = 10
epoches = 20

# network parameters
n_input = 784
n_classes = 10
dropout = 0.75

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = convnet(x, weights=weights, biases=biases, dropout=keep_prob)
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
    train_y = np.asarray(train_y, dtype=np.float32)
    while epoch < epoches:
        step = 1
        acc = 0.0
        loss = 0.0
        while step * batch_size < train_x.shape[0]:
            batch_x = train_x[(step-1)*batch_size:step*batch_size]
            batch_y = train_y[(step - 1) * batch_size:step * batch_size]
            sess.run(optimazer, feed_dict={x:batch_x, y:batch_y, keep_prob:dropout})

            acc += sess.run(accuracy, feed_dict={x:batch_x, y:batch_y, keep_prob:1.0})
            loss += sess.run(cost, feed_dict={x:batch_x, y:batch_y, keep_prob:1.0})
            step += 1
        print("Iter " + str(epoch) + ", Minibatch Loss= " + \
              "{:.6f}".format(loss/(step)) + ", Training Accuracy= " + \
              "{:.5f}".format(acc/(step)))
        epoch += 1
    print('optimazation finished!')

    test_len = 128
    test_data, test_labels = load_mnist(one_hot=True, train=False)
    print('Testing accuracy:', sess.run(accuracy, feed_dict={x:test_data, y:test_labels, keep_prob:1.0}))




