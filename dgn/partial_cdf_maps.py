# -*- coding: utf-8 -*-
"""
Maps corresponding to cumulative density function over part of range and smooth
extension elsewhere, allowing mapping of density corresponding to CDF to
uniform density over some range.
"""

__authors__ = 'Matt Graham'
__license__ = 'MIT'

import numpy as np
import theano as th
import theano.tensor as tt
from scipy.special import erf, erfinv


def norm_pdf(x):
    return np.exp(-0.5 * x**2) / (2 * np.pi)**0.5


def norm_pdf_deriv(x):
    return - x * np.exp(-0.5 * x**2) / (2 * np.pi)**0.5


class PartialGaussianCdfMap(object):

    def __init__(self, b):
        self.b = b
        self.c = erfinv(self.b) * 2.**0.5
        d = norm_pdf(self.c)
        e = norm_pdf_deriv(self.c)
        self.alpha = - e / (8 * d**3)
        self.beta = 0.5 / d - 2 * self.alpha * self.b
        self.gamma = self.c - self.beta * self.b - self.alpha * self.b**2

    def inverse_numpy(self, y):
        abs_y = np.abs(y)
        x = np.empty_like(y)
        y_1, y_2 = y[abs_y < self.b], y[abs_y >= self.b]
        x[abs_y < self.b] = erfinv(y_1) * 2.**0.5
        x[abs_y >= self.b] = ((self.alpha * y_2**2 + self.gamma) * np.sign(y_2)
                              + self.beta * y_2)
        return x

    def forward_numpy(self, x):
        abs_x = np.abs(x)
        y = np.empty_like(x)
        y[abs_x < self.c] = erf(x[abs_x < self.c] / 2.**0.5)
        y[abs_x >= self.c] = (((self.beta**2 - 4 * self.alpha *
                                (self.gamma - abs_x[abs_x >= self.c]))**0.5
                              - self.beta) /
                              (2 * self.alpha)) * np.sign(x[abs_x >= self.c])
        return y

    def inverse_deriv_numpy(self, y):
        abs_y = np.abs(y)
        dx = np.empty_like(y)
        dx[abs_y < self.b] = 0.5 / norm_pdf(erfinv(y[abs_y < self.b]) * 2**0.5)
        dx[abs_y >= self.b] = (2 * self.alpha * abs_y[abs_y >= self.b]
                               + self.beta)
        return dx

    def forward_deriv_numpy(self, x):
        abs_x = np.abs(x)
        dy = np.empty_like(x)
        dy[abs_x < self.c] = 2 * norm_pdf(x[abs_x < self.c])
        dy[abs_x >= self.c] = (self.beta**2 - 4 * self.alpha *
                               (self.gamma - abs_x[abs_x >= self.c]))**(-0.5)
        return dy

    def inverse_theano(self, y):
        x = tt.switch(tt.abs_(y) < self.b, tt.erfinv(y) * 2.**0.5,
                      (self.alpha * y**2 + self.gamma) * tt.sgn(y)
                      + self.beta * y)
        return x

    def forward_theano(self, x):
        abs_x = tt.abs_(x)
        y = tt.switch(abs_x < self.c, tt.erf(x / 2.**0.5),
                      (((self.beta**2 - 4 * self.alpha *
                        (self.gamma - abs_x))**0.5
                       - self.beta) /
                       (2 * self.alpha)) * tt.sgn(x))
        return y

    def mapped_log_density_numpy(self, y):
        n_in_bounds = (np.abs(y) < self.b).sum()
        n_dim = y.shape[0]
        x = self.inverse_numpy(y[np.abs(y) >= self.b])
        return ((n_in_bounds - n_dim) * np.log(np.pi * 2) / 2. -
                n_in_bounds * np.log(2) +
                0.5 * (np.log(self.beta**2 - 4 * self.alpha *
                       (self.gamma - np.abs(x))) - x**2).sum()
                )

    def mapped_log_density_theano(self, y):
        n_in_bounds = (tt.abs_(y) < self.b).sum()
        n_dim = y.shape[0]
        x = self.inverse_theano(y[(tt.abs_(y) >= self.b).nonzero()])
        return ((n_in_bounds - n_dim) * tt.log(np.pi * 2) / 2. -
                n_in_bounds * tt.log(2) +
                0.5 * (tt.log(self.beta**2 - 4 * self.alpha *
                       (self.gamma - tt.abs_(x))) - x**2).sum()
                )


class PartialLogisticCdfMap(object):

    def __init__(self, b):
        self.b = b
        self.c = 2 * np.arctanh(b)
        self.alpha = 2 * b / (b**2 - 1)**2
        self.beta = (2 / (1 - b**2)) - 2 * alpha * b
        self.gamma = self.c - alpha * b**2 - beta * b

    def inverse_numpy(self, y):
        abs_y = np.abs(y)
        x = np.empty_like(y)
        y_1, y_2 = y[abs_y < self.b], y[abs_y >= self.b]
        x[abs_y < self.b] = 2 * np.arctanh(y_1)
        x[abs_y >= self.b] = ((self.alpha * y_2**2 + self.gamma) * np.sign(y_2)
                              + self.beta * y_2)
        return x

    def forward_numpy(self, x):
        abs_x = np.abs(x)
        y = np.empty_like(x)
        y[abs_x < self.c] = np.tanh(x[abs_x < self.c] / 2.)
        y[abs_x >= self.c] = (((self.beta**2 - 4 * self.alpha *
                                (self.gamma - abs_x[abs_x >= self.c]))**0.5
                              - self.beta) /
                              (2 * self. alpha)) * np.sign(x[abs_x >= self.c])
        return y
