# -*- coding: utf-8 -*-
# author = sai
'''
python 版本　DCA
'''
import numpy as np
from collections import OrderedDict

def dca(X):
    artSbx = np.dot(X.T, X)
    eigvals, eigvecs = np.linalg.eig(artSbx)
    eigvals = np.abs(eigvals)

    # ignore zero eigenvalues
    max_eigval = np.max(eigvals)
    nonzero_eigindex = np.where(eigvals / max_eigval >= 1e-6)

    eigvals = eigvals[nonzero_eigindex[0]]
    eigvecs = eigvecs[:, nonzero_eigindex[0]]

    # between-class scatter matrix (Sbx)
    Sbx_eigvecs = X.dot(eigvecs)
    norm_sbx = np.linalg.norm(Sbx_eigvecs, axis=0)

    # normalize
    Sbx_eigvecs = Sbx_eigvecs / norm_sbx

    Sbx_eigvals = np.diag(eigvals)
    Wbx = Sbx_eigvecs.dot(Sbx_eigvals ** (-1 / 2))
    return Wbx, len(eigvals)

def DcaFuse(X, Y, label):
    '''
    :param X: pxn maatiex containing the first set of training feature vectors
    :param Y: qxn matirx containing the second set of training feature vectors
    :param label: 1xn row vector of length n containing the class labels
    :return:
            Ax: Transformation matirx for the first data set (rxp)
                r: maximum dimensionality in the new subspace
            Ay: Transformation matrix for the second data set (rxq)
            Xs: First set of transformed feature vectors (rxn)
            Xy: Second set of transformed feature vectors (rxn)

    '''
    p, n = X.shape
    if Y.shape[1] is not n:
        print 'X and Y must have the same number of columns (samples)'
        return 0
    elif len(label) is not n:
        print 'The length of the label must be equal to the number of samples'
        return 0
    elif n == 1:
        print 'X and Y must have more than one column (samples)'
        return 0
    q = Y.shape[1]
    classes = np.unique(label)
    c = len(classes)

    cellX = OrderedDict()
    cellY = OrderedDict()
    n_sample = OrderedDict()

    # 由于标签混乱，此处为寻找同类数据
    for i in xrange(c):
        index = np.where(label == classes[i])
        n_sample[i] = len(index[0])
        cellX[i] = X[:, index]
        cellY[i] = Y[:, index]

    # Mean of all training data in X
    meanX = np.mean(X)
    meanY = np.mean(Y)

    class_mean_x = np.zeros((p, c))
    class_mean_y = np.zeros((q, c))
    for i in xrange(c):
        class_mean_x[:, i] = np.mean(cellX[i])
        class_mean_y[:, i] = np.mean(cellY[i])

    phibX = np.zeros((p, c))
    phibY = np.zeros((q, c))
    for i in xrange(c):
        phibX[:, i] = np.sqrt(n_sample[i]) * (class_mean_x[:, i] - meanX)
        phibY[:, i] = np.sqrt(n_sample[i]) * (class_mean_y[:, i] - meanY)

    # DCA过程
    wbx, cx = dca(phibX)
    wby, cy = dca(phibY)

    # project data in a space, where the between-class scatter matrices are \
    # identity and the classes are separated
    r = np.min(cx, cy)
    wbx = wbx[:, :r]
    wby = wby[:, :r]
    Xp = wbx.T.dot(X)
    Yp = wby.T.dot(Y)

    # unitize the between-set convariance matirx (Sxy)
    Sxy = Xp.dot(Yp.T)
    wcx, S, wcy = np.linalg.svd(Sxy)

    wcx = wcx.dot(np.diag(S)**(-1/2))
    wcy = wcy.dot(np.diag(S) ** (-1 / 2))

    Xs = wcx.T.dot(Xp)
    Ys = wcy.T.dot(Yp)

    Ax = wcx.T.dot(wbx.T)
    Ay = wcy.T.dot(wby.T)
    return Xs, Ys, Ax, Ay