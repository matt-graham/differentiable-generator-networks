# -*- coding: utf-8 -*-
"""Invertible density network layer definitions."""

__authors__ = 'Matt Graham'
__license__ = 'MIT'

import numpy as np
import theano as th
import theano.tensor as tt
import theano.tensor.slinalg as slinalg
from theano_cpu_ops import (
    log_det, lower_triangular_solve, upper_triangular_solve)


class DensityNetworkLayer(object):
    """ Base class for invertible density network layers. """

    def __init__(self, params):
        self.params = params

    def param_log_prior(self):
        return tt.constant(0.)

    def forward_map(self, x):
        raise NotImplementedError()

    def inverse_map(self, y):
        raise NotImplementedError()

    def forward_jacobian_log_det(self, x):
        raise NotImplementedError()

    def compile_theano_functions(self):
        """ Compile functions from symbolic theano methods defined in class.

        Intended only to be used for unit testing of methods therefore not
        called by default during construction of object as generally a
        whole symbolic computational graph should be compiled from the
        composition of multiple layers rather than compiling functions for
        each layer separately.
        """
        x_batch = tt.matrix('x_batch')
        x_point = tt.vector('x_point')
        y_batch = tt.matrix('y_batch')
        y_point = tt.vector('y_point')
        self.forward_map_batch = th.function(
            inputs=[x_batch],
            outputs=self.forward_map(x_batch)
        )
        self.forward_map_point = th.function(
            inputs=[x_point],
            outputs=self.forward_map(x_point)
        )
        self.inverse_map_batch = th.function(
            inputs=[y_batch],
            outputs=self.inverse_map(y_batch)
        )
        self.inverse_map_point = th.function(
            inputs=[y_point],
            outputs=self.inverse_map(y_point)
        )
        self.forward_jacobian_log_det_batch = th.function(
            inputs=[x_batch],
            outputs=self.forward_jacobian_log_det(x_batch),
            on_unused_input='ignore'
        )
        self.forward_jacobian_log_det_point = th.function(
            inputs=[x_point],
            outputs=self.forward_jacobian_log_det(x_point),
            on_unused_input='ignore'
        )


class LeapfrogLayer(DensityNetworkLayer):
    """
    Layer applying invertible iterated leapfrog type transformation.
    """

    def __init__(self, map_1, map_2, split, n_iter=1):
        self.map_1 = map_1
        self.map_2 = map_2
        self.split = split
        self.n_iter = n_iter
        super(LeapfrogLayer, self).__init__(
            self.map_1.params + self.map_2.params)

    def param_log_prior(self):
        return self.map_1.param_log_prior() + self.map_2.param_log_prior()

    def forward_map(self, x):
        if x.ndim == 1:
            x = x.reshape((1, -1))
            n_dim_orig = 1
        elif x.ndim == 2:
            n_dim_orig = 2
        else:
            raise ValueError('x must be one or two dimensional.')
        x1, x2 = x[:, :self.split], x[:, self.split:]
        for s in range(self.n_iter):
            y1 = x1 + self.map_1(x2)
            y2 = x2 + self.map_2(y1)
            x1, x2 = y1, y2
        y = tt.join(1, y1, y2)
        if n_dim_orig == 1:
            y = y.flatten()
        return y

    def inverse_map(self, y):
        if y.ndim == 1:
            y = y.reshape((1, -1))
            n_dim_orig = 1
        elif y.ndim == 2:
            n_dim_orig = 2
        else:
            raise ValueError('y must be one or two dimensional.')
        y1, y2 = y[:, :self.split], y[:, self.split:]
        for s in range(self.n_iter):
            x2 = y2 - self.map_2(y1)
            x1 = y1 - self.map_1(x2)
            y1, y2 = x1, x2
        x = tt.join(1, x1, x2)
        if n_dim_orig == 1:
            x = x.flatten()
        return x

    def forward_jacobian_log_det(self, x):
        return tt.constant(0.)


class AffineLayer(DensityNetworkLayer):
    """
    Layer applying general affine transformation.

    Forward map: x -> W.dot(x) + b
    """

    def __init__(self, weights_init, biases_init, weights_prec=0.,
                 biases_prec=0., weights_mean=None, biases_mean=None):
        assert weights_init.ndim == 2, 'weights_init must be 2D array.'
        assert biases_init.ndim == 1, 'biases_init must be 1D array.'
        assert weights_init.shape[0] == biases_init.shape[0], \
            'Dimensions of weights_init and biases_init must be consistent.'
        self.weights = th.shared(weights_init, name='W')
        self.biases = th.shared(biases_init, name='b')
        self.weights_prec = weights_prec
        self.biases_prec = biases_prec
        if weights_mean is None:
            weights_mean = np.identity(weights_init.shape[0])
        if biases_mean is None:
            biases_mean = np.zeros_like(biases_init)
        self.weights_mean = weights_mean
        self.biases_mean = biases_mean
        super(AffineLayer, self).__init__([self.weights, self.biases])

    def param_log_prior(self):
        return -(0.5 * self.weights_prec *
                ((self.weights - self.weights_mean)**2).sum() +
                 0.5 * self.biases_prec *
                ((self.biases - self.biases_mean)**2).sum())

    def forward_map(self, x):
        return x.dot(self.weights.T) + self.biases

    def inverse_map(self, y):
        return slinalg.solve(self.weights, (y - self.biases).T).T

    def forward_jacobian_log_det(self, x):
        if x.ndim == 1:
            return log_det(self.weights)
        elif x.ndim == 2:
            return x.shape[0] * log_det(self.weights)
        else:
            raise ValueError('x must be one or two dimensional.')


class DiagonalAffineLayer(DensityNetworkLayer):
    """
    Layer applying restricted affine transformation.

    Matrix restricted to diagonal transformation.

    Forward map: x -> diag(d).dot(x) + b
    """

    def __init__(self, diag_weights_init, biases_init,
                 diag_weights_prec=0., biases_prec=0.,
                 diag_weights_mean=None, biases_mean=None):
        assert diag_weights_init.ndim == 1, (
            'diag_weights_init must be 1D array.')
        assert biases_init.ndim == 1, 'biases_init must be 1D array.'
        assert diag_weights_init.size == biases_init.size, (
            'Dimensions of diag_weights_init and biases_init inconsistent.')
        self.diag_weights = th.shared(diag_weights_init, name='d')
        self.biases = th.shared(biases_init, name='b')
        self.diag_weights_prec = diag_weights_prec
        self.biases_prec = biases_prec
        if diag_weights_mean is None:
            diag_weights_mean = np.ones_like(diag_weights_init)
        if biases_mean is None:
            biases_mean = np.zeros_like(biases_init)
        self.diag_weights_mean = diag_weights_mean
        self.biases_mean = biases_mean
        super(DiagonalAffineLayer, self).__init__(
            [self.diag_weights, self.biases])

    def param_log_prior(self):
        return -(0.5 * self.diag_weights_prec *
                ((self.diag_weights - self.diag_weights_mean)**2).sum() +
                 0.5 * self.biases_prec *
                ((self.biases - self.biases_mean)**2).sum())

    def forward_map(self, x):
        if x.ndim == 1:
            return x * self.diag_weights + self.biases
        elif x.ndim == 2:
            return x * self.diag_weights + self.biases
        else:
            raise ValueError('x must be one or two dimensional.')

    def inverse_map(self, y):
        return (y - self.biases) / self.diag_weights

    def forward_jacobian_log_det(self, x):
        if x.ndim == 1:
            return tt.log(tt.abs_(self.diag_weights)).sum()
        elif x.ndim == 2:
            return x.shape[0] * tt.log(tt.abs_(self.diag_weights)).sum()
        else:
            raise ValueError('x must be one or two dimensional.')


class DiagPlusRank1AffineLayer(DensityNetworkLayer):
    """
    Layer applying restricted affine transformation.

    Matrix restricted to diagonal plus rank-1 transformation.

    Forward map: x -> (diag(d) + outer(u, v)).dot(x) + b
    """

    def __init__(self, diag_weights_init, u_vect_init, v_vect_init,
                 biases_init, diag_weights_prec=0., u_vect_prec=0.,
                 v_vect_prec=0., biases_prec=0., diag_weights_mean=None,
                 u_vect_mean=None, v_vect_mean=None, biases_mean=None):
        assert diag_weights_init.ndim == 1, (
            'diag_weights_init must be 1D array.')
        assert u_vect_init.ndim == 1, 'u_vect_init must be 1D array.'
        assert v_vect_init.ndim == 1, 'v_vect_init must be 1D array.'
        assert biases_init.ndim == 1, 'biases_init must be 1D array.'
        assert (diag_weights_init.size == u_vect_init.size and
                diag_weights_init.size == v_vect_init.size and
                diag_weights_init.size == biases_init.size), (
            'Dimensions of diag_weights_init, u_vect_unit,'
            ' v_vect_init and biases_init inconsistent.')
        self.diag_weights = th.shared(diag_weights_init, name='d')
        self.u_vect = th.shared(u_vect_init, name='u')
        self.v_vect = th.shared(v_vect_init, name='v')
        self.biases = th.shared(biases_init, name='b')
        self.diag_weights_prec = diag_weights_prec
        self.u_vect_prec = u_vect_prec
        self.v_vect_prec = v_vect_prec
        self.biases_prec = biases_prec
        if diag_weights_mean is None:
            diag_weights_mean = np.ones_like(diag_weights_init)
        if u_vect_mean is None:
            u_vect_mean = np.zeros_like(u_vect_init)
        if v_vect_mean is None:
            v_vect_mean = np.zeros_like(v_vect_init)
        if biases_mean is None:
            biases_mean = np.zeros_like(biases_init)
        self.diag_weights_mean = diag_weights_mean
        self.u_vect_mean = u_vect_mean
        self.v_vect_mean = v_vect_mean
        self.biases_mean = biases_mean
        super(DiagPlusRank1AffineLayer, self).__init__(
            [self.diag_weights, self.u_vect, self.v_vect, self.biases])

    def param_log_prior(self):
        return -(0.5 * self.diag_weights_prec *
                ((self.diag_weights - self.diag_weights_mean)**2).sum() +
                 0.5 * self.u_vect_prec *
                ((self.u_vect - self.u_vect_mean)**2).sum() +
                 0.5 * self.v_vect_prec *
                ((self.v_vect - self.v_vect_mean)**2).sum() +
                 0.5 * self.biases_prec *
                ((self.biases - self.biases_mean)**2).sum())

    def forward_map(self, x):
        if x.ndim == 1:
            return (x * self.diag_weights + self.u_vect * x.dot(self.v_vect)
                    + self.biases)
        elif x.ndim == 2:
            return (x * self.diag_weights +
                    self.u_vect[None, :] * (x.dot(self.v_vect)[:, None]) +
                    self.biases)
        else:
            raise ValueError('x must be one or two dimensional.')

    def inverse_map(self, y):
        z = (y - self.biases) / self.diag_weights
        u_vect_over_diag_weights = (self.u_vect / self.diag_weights)
        if y.ndim == 1:
            return (z - u_vect_over_diag_weights *
                    (z.dot(self.v_vect)) /
                    (1 + self.v_vect.dot(u_vect_over_diag_weights)))
        elif y.ndim == 2:
            return (z - u_vect_over_diag_weights[None, :] *
                    (z.dot(self.v_vect))[:, None] /
                    (1 + self.v_vect.dot(u_vect_over_diag_weights)))
        else:
            raise ValueError('y must be one or two dimensional.')

    def forward_jacobian_log_det(self, x):
        if x.ndim == 1:
            return (tt.log(tt.abs_(1 + self.v_vect.dot(self.u_vect /
                                                       self.diag_weights))) +
                    tt.log(tt.abs_(self.diag_weights)).sum())
        elif x.ndim == 2:
            return x.shape[0] * (
                tt.log(tt.abs_(1 + self.v_vect.dot(self.u_vect /
                                                   self.diag_weights))) +
                tt.log(tt.abs_(self.diag_weights)).sum()
            )
        else:
            raise ValueError('x must be one or two dimensional.')


class TriangularAffineLayer(DensityNetworkLayer):
    """
    Layer applying restricted affine transformation.

    Matrix restricted to be triangular.

    Forward map:
    if lower:
        x -> tril(W).dot(x) + b
    else:
        x -> triu(W).dot(x) + b
    """

    def __init__(self, weights_init, biases_init, lower=False,
                 weights_prec=0., biases_prec=0., weights_mean=None,
                 biases_mean=None):
        assert weights_init.ndim == 2, 'weights_init must be 2D array.'
        assert biases_init.ndim == 1, 'biases_init must be 1D array.'
        assert weights_init.shape[0] == biases_init.shape[0], \
            'Dimensions of weights_init and biases_init must be consistent.'
        self.lower = lower
        self.weights = th.shared(weights_init, name='W')
        self.weights_tri = (tt.tril(self.weights)
                            if lower else tt.triu(self.weights))
        self.biases = th.shared(biases_init, name='b')
        self.weights_prec = weights_prec
        self.biases_prec = biases_prec
        if weights_mean is None:
            weights_mean = np.eye(weights_init.shape[0])
        if biases_mean is None:
            biases_mean = np.zeros_like(biases_init)
        self.weights_mean = (np.tril(weights_mean)
                             if lower else np.triu(weights_mean))
        self.biases_mean = biases_mean
        super(TriangularAffineLayer, self).__init__(
            [self.weights, self.biases])

    def param_log_prior(self):
        return -(0.5 * self.weights_prec *
                 ((self.weights_tri - self.weights_mean)**2).sum()
                 + 0.5 * self.biases_prec *
                 ((self.biases - self.biases_mean)**2).sum())

    def forward_map(self, x):
        return x.dot(self.weights_tri.T) + self.biases

    def inverse_map(self, y):
        if self.lower:
            return lower_triangular_solve(self.weights_tri,
                                          (y - self.biases).T).T
        else:
            return upper_triangular_solve(self.weights_tri,
                                          (y - self.biases).T).T

    def forward_jacobian_log_det(self, x):
        if x.ndim == 1:
            return tt.log(tt.abs_(tt.nlinalg.diag(self.weights))).sum()
        elif x.ndim == 2:
            return (x.shape[0] *
                    tt.log(tt.abs_(tt.nlinalg.diag(self.weights))).sum())
        else:
            raise ValueError('x must be one or two dimensional.')


class ElementwiseLayer(DensityNetworkLayer):
    """
    Layer applying bijective elementwise transformation.

    Forward map: x -> f(x)
    """

    def __init__(self, forward_func, inverse_func, fudge=0.):
        self.forward_func = forward_func
        self.inverse_func = inverse_func
        self.fudge = fudge
        super(ElementwiseLayer, self).__init__([])

    def forward_map(self, x):
        return self.forward_func(x)

    def inverse_map(self, y):
        return self.inverse_func(y)

    def forward_jacobian_log_det(self, x):
        dy_dx, _ = th.scan(lambda x_i: th.grad(self.forward_func(x_i), x_i),
                           sequences=[x.flatten()])
        if self.fudge != 0.:
            return tt.log(dy_dx + self.fudge).sum()
        else:
            return tt.log(dy_dx).sum()


class FwdDiagInvElementwiseLayer(DiagonalAffineLayer):
    """
    Layer applying forward elementwise map, diagonal scaling then inverse map.

    Forward map: x -> f(d * g(x)) where g(f(x)) = f(g(x)) = x
    """

    def __init__(self, forward_func, inverse_func,
                 diag_weights_init, biases_init,
                 diag_weights_prec=0., biases_prec=0.,
                 diag_weights_mean=None, biases_mean=None,
                 fudge=0.):
        self.forward_func = forward_func
        self.inverse_func = inverse_func
        self.fudge = fudge
        super(FwdDiagInvElementwiseLayer, self).__init__(
            diag_weights_init, biases_init, diag_weights_prec,
            biases_prec, diag_weights_mean, biases_mean)

    def forward_map(self, x):
        return self.forward_func(self.diag_weights * self.inverse_func(x) +
                                 self.biases)

    def inverse_map(self, y):
        return self.forward_func((self.inverse_func(y) - self.biases) /
                                 self.diag_weights)

    def forward_jacobian_log_det(self, x):
        y_sum = self.forward_map(x).sum()
        dy_dx = th.grad(y_sum, x)
        if self.fudge != 0.:
            return tt.log(dy_dx + self.fudge).sum()
        else:
            return tt.log(dy_dx).sum()


class PermuteDimensionsLayer(DensityNetworkLayer):
    """
    Layer applying permutation of dimensions.

    Forward map: x -> x[perm]
    """

    def __init__(self, perm):
        self.perm = th.shared(perm, name='perm')
        super(PermuteDimensionsLayer, self).__init__([])

    def forward_map(self, x):
        return tt.permute_row_elements(x, self.perm)

    def inverse_map(self, y):
        return tt.permute_row_elements(y, self.perm, inverse=True)

    def forward_jacobian_log_det(self, x):
        return tt.constant(0.)
