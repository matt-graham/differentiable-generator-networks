# -*- coding: utf-8 -*-
"""
Additional Theano ops
"""

__authors__ = 'Matt Graham'
__copyright__ = 'Copyright 2015, Matt Graham'
__license__ = 'MIT'

import numpy as np
import theano as th
import theano.tensor as tt
from theano.tensor import slinalg


class LogDet(th.Op):
    """
    Matrix log (absolute) determinant. Input should be a square matrix.
    """

    __props__ = ()

    def make_node(self, x):
        x = tt.as_tensor_variable(x)
        assert x.ndim == 2
        o = th.tensor.scalar(dtype=x.dtype)
        return th.Apply(self, [x], [o])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        try:
            s, ld = np.linalg.slogdet(x)
            z[0] = np.asarray(ld, dtype=x.dtype)
        except np.linalg.LinAlgError:
            print('Failed to compute log determinant of {0}'.format(x))
            raise

    def grad(self, inputs, g_outputs):
        gz, = g_outputs
        x, = inputs
        return [slinalg.solve(x.T, gz)]

    def infer_shape(self, node, shapes):
        return [()]

    def __str__(self):
        return "LogDet"

log_det = LogDet()
