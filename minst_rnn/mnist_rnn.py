# -*- coding: utf-8 -*-
# author = sai
import numpy
import cPickle
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Activation, Convolution2D, ZeroPadding2D, TimeDistributed, Lambda
from keras.optimizers import RMSprop
def to_categorical(y, nb_classes=None):
    if not nb_classes:
        nb_classes = numpy.max(y)+1
    Y = numpy.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

def cnn2rnn(x):
    shape = x.shape
    time_steps = shape[1] * shape[2]
    input_dim = shape[3]
    return x.reshape((shape[0], time_steps, input_dim))

def output_shape(input_shape):
    time_steps = input_shape[1] * input_shape[2]
    input_dim = input_shape[3]
    return (None, time_steps, input_dim)

with open('mnist.pkl', 'r') as f:
    _train, _val, _test = cPickle.load(f)
train_x, train_y = _train
val_x, val_y = _val
train_y = to_categorical(train_y, nb_classes=10)
val_y = to_categorical(val_y, nb_classes=10)
train_x = train_x.reshape((50000, 1, 28, 28))/255.
val_x = val_x.reshape((10000, 1, 28, 28))/255.
map2step = 8*28
train_y = numpy.tile(train_y, map2step).reshape(train_y.shape[0], map2step, 10)
val_y = numpy.tile(val_y, map2step).reshape(val_y.shape[0], map2step, 10)
shape = train_x.shape[1:]

# model
model = Sequential()
model.add(ZeroPadding2D(padding=(1, 1), input_shape=(1, 28, 28)))
model.add(Convolution2D(8, 3, 3))
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(8, 3, 3))
model.add(Lambda(cnn2rnn, output_shape=output_shape))
model.add(SimpleRNN(input_shape=shape, output_dim=50, return_sequences=True))
# model.add(Dense(input_dim=50, output_dim=10))
model.add(TimeDistributed(Dense(output_dim=10)))
model.add(Activation('softmax'))
rms = RMSprop(lr=0.003)
model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_x, train_y, nb_epoch=200, batch_size=100, verbose=1, validation_data=(val_x, val_y))