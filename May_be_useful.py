'''
There are some functions which I wrote as you may want to add them into the raw structure
'''
import tensorflow as tf
import numpy as np


# Activation functions
def parametric_relu(x, alpha=0.01, max_value=None):
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=_FLOATX),
                             tf.cast(max_value, dtype=_FLOATX))
    x -= tf.constant(alpha) * negative_part
    return x

def max_out(inputs, outputs):
    return tf.reduce_max([inputs, outputs], axis=0)


# PCA and whitening and batch normalization
def pca_zm_proj(X, K=None):
    if np.max(np.abs(np.mean(X,0))) > 1e-5:
        raise ValueError('Data is not zero mean.')
    if K is None:
        K = X.shape[1]
    E, V = np.linalg.eig(np.dot(X.T, X))
    idx = np.argsort(E)[::-1]
    V = V[:, idx[:K]] # D,K
    return V

def whiten_SVD(X):
    U, s, Vt = np.linalg.svd(X)
    X_white = np.dot(U, Vt)
    return X_white

def whiten(X,fudge=1E-18):
    Xcov = np.dot(X.T,X)
    d, V = np.linalg.eigh(Xcov)
    D = np.diag(1. / np.sqrt(d+fudge))
    W = np.dot(np.dot(V, D), V.T)
    return W

# tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name=None)

