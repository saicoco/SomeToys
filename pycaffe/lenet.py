# -*- coding: utf-8 -*-
# author = sai

from __future__ import print_function
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import numpy as np

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
weight_filler = dict(type='gaussian', std=0.01)
bias_filler=dict(type='constant', value=0.1)
param = [weight_param, bias_param]

'''
关于其中参数的参考，来自于文件caffe_pb2,对于系统默认参数，使用P.XXX.param,其余则通过对应赋值来实现
'''
def conv(inputs, kernel_size,  num_output, stride=1, pad=0, param=param, weight_filler=weight_filler, bias_filler=bias_filler):
    conv = L.Convolution(inputs, kernel_size=5, pad=pad, stride=stride, num_output=num_output, param=param[0], weight_filler=weight_filler, bias_term=False)
    bn_conv = L.BatchNorm(conv, use_global_stats=False)
    scale_bn = L.Scale(bn_conv, bias_term=True, bias_filler=bias_filler)
    return scale_bn, L.ReLU(scale_bn, in_place=True)

def pool(inputs, kernel_size=2, stride=2):
    pool = L.Pooling(inputs, kernel_size=kernel_size, stride=stride, pool=P.Pooling.MAX)
    return pool

def FC(inputs,  num_output, param=param, weight_filler=weight_filler, bias_filler=bias_filler):
    fc = L.InnerProduct(inputs, num_output= num_output, param=param, weight_filler=weight_filler, bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def lenet(lmdb, batch_size=128):

    data, labels = L.Data(source=lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2)
    conv1, relu1 = conv(data, kernel_size=5, num_output=20)
    pool1 = pool(relu1)
    conv2, relu2 = conv(pool1, kernel_size=5, num_output=50)
    pool2 = pool(relu2)
    fc1, relu3 = FC(pool2, num_output=500)
    loss = L.SoftmaxWithLoss(fc1, labels)
    acc = L.Accuracy(fc1, labels)
    return to_proto(loss, acc)

def make_net(train_pt, test_pt):
    with open(train_pt, 'w') as f:
        print(lenet('/home/sai/caffe/examples/mnist/mnist_train_lmdb'), file=f)

    with open(test_pt, 'w') as f:
        print(lenet('/home/sai/caffe/examples/mnist/mnist_test_lmdb', batch_size=50), file=f)

def solver(train_pt, test_pt=None, base_lr=0.001, max_iter=2000):
    s = caffe_pb2.SolverParameter()
    s.train_net = train_pt
    if test_pt is not None:
        s.test_net.append(test_pt)
        s.test_interval = 20
        s.test_iter.append(100)
    s.base_lr = base_lr
    s.max_iter = max_iter
    s.lr_policy = 'step'
    s.stepsize = 500
    s.snapshot = 1000
    s.snapshot_prefix = './snapshot/lenet'
    s.momentum = 0.9
    s.weight_decay = 5e-4
    s.display = 20
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    with open('lenet_solver.prototxt', 'w') as f:
        f.write(str(s))
    print(f.name)
    return f.name

def run_solvers(epoches, solvers, dis = 20):

    blobs = ('SoftmaxWithLoss1', 'Accuracy1')
    loss, acc = (np.zeros(epoches) for blob in blobs)
    print(solvers.net.blobs)
    for ite in range(epoches):
        solvers.step(1)
        loss[ite], acc[ite] = (solvers.net.blobs[blob].data.copy() for blob in blobs)
        if ite == dis:
            loss_disp = '; '.join('loss=%.3f, acc=%2d%%' %
                                  (loss[ite], np.round(100 * acc[ite]))
                                  )
            print('ite = {}, {}'.format(ite, loss_disp))
    return loss, acc



if __name__=='__main__':
    train_pt = 'train.prototxt'
    test_pt = 'test.prototxt'
    make_net(train_pt, test_pt)

    epoches = 2000
    lr = 0.001
    solver = solver(train_pt=train_pt,  test_pt=test_pt, max_iter=epoches, base_lr=lr)
    lenet_solvers = caffe.get_solver(solver)
    loss, acc = run_solvers(epoches, lenet_solvers)
    print('Done')
    del solver
