# -*- coding: utf-8 -*-
# author = sai

import caffe
import yaml
import tool
import multiprocessing
import threading
import Queue as queue
import time
import numpy as np

class audiovisualLayer(caffe.Layer):

    def setup(self, bottom, top):
        params_str = self.param_str.split(',')
        params = [yaml.load(item) for item in params_str]
        self.batch_size = params[0]['batch_size']
        self.data_dir = params[1]['data_dir']
        self.phase = params[2]['train']
        # avgen = tool.av_generator(batch_size=self.batch_size, data_dir=self.data_dir, train=self.phase)
        # self.batch_loader, _, _ = generator_queue(avgen, nb_worker=4, pickle_safe=True)
        self.batch_loader = tool.av_generator(batch_size=self.batch_size, data_dir=self.data_dir, train=self.phase)
        print 'Parameter batch_size:{}\n' \
              'data_dir:{}' \
              'phase:{}'.format(self.batch_size, self.data_dir, self.phase)
        # top: image, audio, label
        top[0].reshape(self.batch_size, 3, 50, 50)
        top[1].reshape(self.batch_size, 375)
        top[2].reshape(self.batch_size)
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        # for i in xrange(self.batch_size):
            # top[0].data[i, ...], top[1].data[i, ...], top[2].data[i, ...] = self.batch_loader.next()
        top[0].data[...], top[1].data[...], top[2].data[...] = self.batch_loader.next()

    def backwrad(self, bottom, propagate_down, top):
        pass

def generator_queue(generator, max_q_size=10,
                    wait_time=0.05, nb_worker=1, pickle_safe=False):

    generator_threads = []
    if pickle_safe:
        q = multiprocessing.Queue(maxsize=max_q_size)
        _stop = multiprocessing.Event()
    else:
        q = queue.Queue()
        _stop = threading.Event()

    try:
        def data_generator_task():
            while not _stop.is_set():
                try:
                    if pickle_safe or q.qsize() < max_q_size:
                        generator_output = next(generator)
                        q.put(generator_output)
                    else:
                        time.sleep(wait_time)
                except Exception:
                    _stop.set()
                    raise

        for i in range(nb_worker):
            if pickle_safe:
                # Reset random seed else all children processes share the same seed
                np.random.seed()
                thread = multiprocessing.Process(target=data_generator_task)
            else:
                thread = threading.Thread(target=data_generator_task)
            generator_threads.append(thread)
            thread.daemon = True
            thread.start()
    except:
        _stop.set()
        if pickle_safe:
            # Terminate all daemon processes
            for p in generator_threads:
                if p.is_alive():
                    p.terminate()
            q.close()
        raise

    return q, _stop, generator_threads

