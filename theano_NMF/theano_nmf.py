# -*- coding: utf-8 -*-
# author = sai

import theano
import theano.tensor as T
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from scipy.io import loadmat
import sys
floatX = theano.config.floatX

def evaluate_nmi(X):
    pred = kmeans.fit_predict(X)
    score = sklearn.metrics.normalized_mutual_info_score(gnd, pred)
    return score
    
def nmf(shape, n_components=400):
    """
    利用theano加速
    """
    X = T.matrix('x')
    Z = theano.shared(np.asarray(0.08 * np.random.rand(shape[0], n_components), dtype=floatX), name='Z', borrow=True)
    H = theano.shared(np.asarray(0.08 * np.random.rand(n_components, shape[1]), dtype=floatX), name='H', borrow=True)
    
    # update H
    z_x = T.dot(Z.T, X)
    zz_h = T.dot(T.dot(Z.T, Z), H)
    H_NEW = H * z_x/zz_h
    # update Z
    x_h = T.dot(X, H_NEW.T)
    z_hh = T.dot(Z, T.dot(H_NEW, H_NEW.T))
    Z_NEW = Z * x_h/z_hh
    updates = [(H, H_NEW), (Z, Z_NEW)]
    cost = T.sqrt(T.sum((X - T.dot(Z, H))**2))
    construct = T.dot(Z, H)
    
    nmf_fun = theano.function(inputs=[X], outputs=[cost, construct], updates=updates)
    return nmf_fun

if __name__=='__main__':
    out = sys.stdout
    mat = loadmat('PIE_pose27.mat', struct_as_record=False, squeeze_me=True)
    data, gnd = mat['fea'].astype('float32'), mat['gnd']
    # Normalise each feature to have an l2-norm equal to one.
    data /= np.linalg.norm(data, 2, 1)[:, None] 
    n_classes = np.unique(gnd).shape[0]
    kmeans = KMeans(n_classes, precompute_distances=False)
    shape = data.T.shape
    nmf_fun = nmf(shape)
    for i in xrange(100):
        cost, cons = nmf_fun(data.T)
        out.write('\repch[%d], cost=%f\r' % (i, cost))
        out.flush()
    fea = cons.T # this is the last layers features i.e. h_2
    pred = kmeans.fit_predict(fea)
    score = sklearn.metrics.normalized_mutual_info_score(gnd, pred)
    print("NMI: {:.2f}%".format(100 * score))