# -*- coding: utf-8 -*-
# author = sai
import caffe
import numpy as np
from caffe.proto import caffe_pb2


def solver(train_pt, test_pt=None, base_lr=0.001, max_iter=2000):
    s = caffe_pb2.SolverParameter()
    s.train_net = train_pt
    if test_pt is not None:
        s.test_net.append(test_pt)
        s.test_interval = 100
        s.test_iter.append(10)
    s.base_lr = base_lr
    s.max_iter = max_iter
    s.lr_policy = 'step'
    s.stepsize = 500
    s.snapshot = 500
    s.snapshot_prefix = './snapshot/'
    s.momentum = 0.9
    s.weight_decay = 5e-4
    s.display = 10
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    with open('conv_level_solver.prototxt', 'w') as f:
        f.write(str(s))
    print(f.name)
    return f.name


def run_solvers(epoches, solvers, dis=20):

    blobs = ('loss', 'acc')
    loss, acc = (np.zeros(epoches) for blob in blobs)
    for ite in range(epoches):
        solvers.step(1)
        loss[ite], acc[ite] = (solvers.net.blobs[blob].data.copy()
                               for blob in blobs)
        if ite == dis:
            loss_disp = '; '.join('loss=%.3f, acc=%2d%%' %
                                  (loss[ite], np.round(100 * acc[ite]))
                                  )
            print('ite = {}, {}'.format(ite, loss_disp))
    return loss, acc

if __name__ == '__main__':
    train_pt = 'conv_level.prototxt'
    test_pt = 'test.prototxt'
    epoches = 2000
    lr = 0.001
    solver = solver(
        train_pt=train_pt,
        test_pt=test_pt,
        max_iter=epoches,
        base_lr=lr)
    conv_solvers = caffe.get_solver(solver)
    loss, acc = run_solvers(epoches, conv_solvers)
    print('Done')
    del solver
