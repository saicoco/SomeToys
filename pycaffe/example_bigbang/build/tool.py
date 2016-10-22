# -*- coding: utf-8 -*-
# author = sai

import numpy as np
import h5py
import random
import threading
import cv2
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

def load_audio(labels, count, audio, tags):
    '''
    根据标签生成与人脸数据对应的语音数据
    '''
    # 得到标签相等得行
    idx = [np.where(np.argmax(tags, axis=1) == labels)[0]]
    # print idx
    np.random.seed(count)
    audio_data = np.zeros((375,))
    # rn = np.random.sample(idx[0], 5)
    rn = [np.random.choice(idx[0]) for i in xrange(5)]  # 从item中随机选取5个数，即选取5行
    # print audio[rn].shape
    audio_data = audio[rn].reshape(375)
    return audio_data

@threadsafe_generator
def image_generator(batch_size, data_dir, train=True):
    images = np.zeros((batch_size, 50, 50, 3), dtype=np.float32)
    labels = []
    count = 0
    if train is True:
        path = data_dir + '/train-file-list.txt'
    else:
        path = data_dir + '/test-file-list.txt'
    with open(path, 'r') as f:
        readlines = f.readlines()
        random.shuffle(readlines)
        while 1:
            for i, item in enumerate(readlines):
                line = item.split(' ')
                label = int(line[1])
                filename = data_dir+line[0]
                img = cv2.imread(filename)
                img = cv2.resize(img, dsize=(50, 50))
                images[count] = img
                labels.append(label)
                count += 1
                if count==batch_size:
                    images = images.transpose(0, 3, 1, 2)/255.
                    # labels = to_categorical(labels)
                    yield (images, labels)
                    count = 0
                    labels = []
                    images = np.zeros((batch_size, 50, 50, 3), dtype=np.float32)

@threadsafe_generator
def av_generator(batch_size, data_dir, train=True):
    images = np.zeros((batch_size, 50, 50, 3), dtype=np.float32)
    audio_data = np.zeros((batch_size, 375))
    labels = []
    count = 0
    if train is True:
        path = data_dir + '/train-file-list.txt'
    else:
        path = data_dir + '/test-file-list.txt'
    with open(path, 'r') as f:
        audio_path = data_dir + '/audio_samples.mat'
        with h5py.File(audio_path, 'r') as ff:
            audio = ff[ff.keys()[0]][:]
            tags = ff[ff.keys()[1]][:]
        readlines = f.readlines()
        random.shuffle(readlines)
        while 1:
            for i, item in enumerate(readlines):
                line = item.split(' ')
                label = int(line[1])
                audio_data[count] = load_audio(label, count, audio, tags)
                filename = data_dir + line[0]
                img = cv2.imread(filename)
                img = cv2.resize(img, dsize=(50, 50))
                images[count] = img
                labels.append(label)
                count += 1
                if count==batch_size:
                    images = images.transpose(0, 3, 1, 2)/255.
                    # labels = to_categorical(labels)
                    yield (images, audio_data, labels)
                    count = 0
                    labels = []
                    images = np.zeros((batch_size, 50, 50, 3), dtype=np.float32)
                    audio_data = np.zeros((batch_size, 375))