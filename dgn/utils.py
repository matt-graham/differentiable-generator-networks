# -*- coding: utf-8 -*-
"""Utility functions for dataset preprocessing and experiment set up."""

__authors__ = 'Matt Graham'
__license__ = 'MIT'


import numpy as np
import scipy.linalg as la
from scipy.fftpack import dct, idct
import logging
import datetime
import os


def dct_2d(x):
    """Perform 2D DCT (type II) on set of images"""
    return dct(dct(x, axis=-2), axis=-1) / (4 * x.shape[-1] * x.shape[-2])**0.5


def idct_2d(x):
    """Perform 2D inverse DCT (type III) on set of images."""
    return (idct(idct(x, axis=-1), axis=-2) /
            (4 * x.shape[-1] * x.shape[-2])**0.5)


def reshape_to_images(x, im_h=None):
    """Reshape set of vectors to images, assuming square or given height."""
    if im_h is None:
        im_h = int(x.shape[1]**0.5)
    im_w = x.shape[1] // im_h
    return x.reshape(x.shape[0], im_h, im_w)


def reshape_to_vectors(x):
    """Reshape set of images to vectors."""
    return x.reshape(x.shape[0], -1)


def logit(x):
    """Calculate logit (inverse logistic sigmoid) of input."""
    return np.log(x) - np.log(1. - x)


def sigmoid(x):
    """Calculate (logistic) sigmoid of input."""
    return 1. / (1. + np.exp(-x))


def normalise(x):
    """Normalise set of vectors by elementwise mean and standard deviation."""
    x_mean = x.mean(0)
    x_std = x.std(0)
    return (x - x_mean) / x_std, x_mean, x_std


def denormalise(x_n, x_mean, x_std):
    """Map from normalised set of vectors to non-normalised originals."""
    return x_n * x_std + x_mean


def whiten(x):
    """Whiten set of vectors using empirical covariance eigendecomposition."""
    x_mean = x.mean(0)
    x_zm = x - x_mean
    M = x_zm.T.dot(x_zm) / x_zm.shape[0]
    e, R = la.eigh(M)
    Q = R / e**0.5
    x_w = x_zm.dot(Q)
    return x_w, x_mean, e, R


def dewhiten(x_w, x_mean, e, R):
    """Map from whitened set of vectors to non-whitened originals."""
    return x_w.dot((R * e**0.5).T) + x_mean


def dequantise(x, prng=np.random, n_levels=256):
    """Dequantise integer values by adding scaled uniform noise."""
    x = x.astype(np.float32) + prng.uniform(size=x.shape)
    x /= float(n_levels)
    return x


def quantise(x, n_levels=256):
    """Map from dequantised values back to integer values by rounding."""
    x = x * n_levels
    x = np.floor(x)
    return x


def rgb_to_yuv(x):
    """Map set of images from RGB colour space to YUV colour space."""
    W = np.array([[0.299, 0.587, 0.114],
                  [-0.16873, -0.33127, 0.5],
                  [0.5, -0.41869, -0.08131]])
    b = np.array([0, 0.5, 0.5])
    if x.ndim == 4:
        x = np.einsum('ij,kjlm->kilm', W, x)
        return x + b[None, :, None, None]
    elif x.ndim == 3:
        x = np.einsum('ij,jlm->ilm', W, x)
        return x + b[:, None, None]
    else:
        raise ValueError('Input must be 3 or 4 dimensional.')


def yuv_to_rgb(x):
    """Map set of images from YUV colour space to RGB colour space."""
    W_inv = np.array([[1., 0., 1.4020],
                      [1., -0.34413, -0.71413],
                      [1., 1.7720, 0.]])
    b = np.array([0, 0.5, 0.5])
    if x.ndim == 4:
        x = x - b[None, :, None, None]
        return np.einsum('ij,kjlm->kilm', W_inv, x)
    elif x.ndim == 3:
        x = x - b[:, None, None]
        return np.einsum('ij,jlm->ilm', W_inv, x)
    else:
        raise ValueError('Input must be 3 or 4 dimensional.')


def shuffle(x, prng=np.random.RandomState()):
    """Randomly permute order of a set."""
    perm = prng.permutation(x.shape[0])
    return x[perm], perm


def deshuffle(x, perm):
    """Return a permuted set to original order."""
    return x[np.argsort(perm)]


def split(x, n_split):
    """Split a set into two parts at a given index."""
    return x[:n_split], x[n_split:]


def generate_random_square_matrices(n_matrix, n_dim, noise_var=0.01, mean=None,
                                    prng=np.random.RandomState()):
    """Generate set of random Gaussian square matrices."""
    mean = np.eye(n_dim) if (mean is None) else mean
    ns = prng.normal(size=(n_matrix, n_dim, n_dim))
    return (ns * noise_var + mean).astype(np.float32)


def generate_random_param_vectors(n_vector, n_dim, noise_var=0.01, mean=0.,
                                  prng=np.random.RandomState()):
    """Generate set of random Gaussian vectors."""
    ns = prng.normal(size=(n_vector, n_dim))
    return (ns * noise_var + mean).astype(np.float32)


def setup_logger(exp_dir, exp_tag):
    """Setup logger for recording experiment details."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    time_stamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    file_handler = logging.FileHandler(
        os.path.join(exp_dir, '{0}_{1}_experiment.log'
                              .format(time_stamp, exp_tag)))
    formatter = logging.Formatter(
        '%(asctime)s %(name)s:%(levelname)s %(message)s', '%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

