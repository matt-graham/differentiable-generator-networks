# -*- coding: utf-8 -*-
"""Standard log densities for use in specifiying base densities."""

__authors__ = 'Matt Graham'
__license__ = 'MIT'

import theano.tensor as tt
import numpy as np


def gaussian_log_density(x):
    return - x.size * np.log(2 * np.pi) / 2. - 0.5 * (x**2).sum()


def logistic_log_density(x):
    return -(tt.log(1 + tt.exp(x)) + tt.log(1 + tt.exp(-x))).sum()
