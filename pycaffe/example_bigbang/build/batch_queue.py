# -*- coding: utf-8 -*-
# author = sai

from multiprocessing import Process, Queue
import tool
import time, random
from py_data_layer import generator_queue

class prefetch_queue(object):

    def __init__(self, batch_size, data_dir, phase=True):
        self.producer = tool.av_generator(batch_size, data_dir, train=phase)
        self.queue = Queue(5)

    def produce(self):
        if not self.queue.full():
            self.queue.put(self.producer.next())
        else:
            pass

    def samples(self):
            if not self.queue.empty():
                item = self.queue.get()
                return item

    def ini_queue(self):
        while not self.queue.full():
            print '....'
            self.produce()

    def next(self):
        self.produce()
        return self.samples()

if __name__ == '__main__':
    # pre = prefetch_queue(128, '../data/BingBang/')
    # pre.ini_queue()
    av = tool.av_generator(128, '../data/BingBang/')
    q, _stop, threads = generator_queue(av, nb_worker=4, pickle_safe=True)
    while True:
        t1 = time.time()
        item = q.get()
        t2 = time.time() - t1

        t3 = time.time()
        item = av.next()
        t4 = time.time() - t3
        print 'queue:{}, avgen:{}'.format(t2, t4)