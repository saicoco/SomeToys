# -*- coding: utf-8 -*-
# author = sai

import caffe
import yaml, glob
from random import shuffle
import cv2
import logging
logging.basicConfig(level=logging.INFO)

class myPythonLayer(caffe.Layer):
    """
    reshape images
    """
    def setup(self, bottom, top):
        params_str = self.param_str.split(',')
        params = [yaml.load(item) for item in params_str]
        print params
        self.source = params[0]['source_dir']
        self.target_size = params[1]['target_size']
        self.batch_size = params[2]['batch_size']
        self.batch_loader = BatchLoader(source_dir=self.source, target_size=self.target_size)
        print 'Parameter batch_size:{}\n' \
              'source_dir:{}\n' \
              'target_size:{}'.format(self.batch_size, self.source, self.target_size)
        top[0].reshape(self.batch_size, self.target_size, self.target_size, 3)
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for i in xrange(self.batch_size):
            top[0].data[i, ...] = self.batch_loader.next_batch()
    def backward(self, bottom, propagate_down, top):
        pass

class BatchLoader(object):

    def __init__(self, source_dir, target_size):
        self.cur = 0
        self.target_size = target_size
        self.indexlist = glob.glob(source_dir+ '/*.jpg')

    def next_batch(self):

        if self.cur == len(self.indexlist):
            self.cur = 0
            shuffle(self.indexlist)
        item = self.indexlist[self.cur]
        img_tmp = cv2.imread(item)
        img_tmp = cv2.resize(src=img_tmp, dsize=(self.target_size, self.target_size))
        self.cur += 1
        logging.info('load {} images'.format(self.cur))
        return img_tmp





