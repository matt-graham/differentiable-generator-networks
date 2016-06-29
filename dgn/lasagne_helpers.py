# -*- coding: utf-8 -*-
"""Helper functions for using Lasagne based networks in models."""

__authors__ = 'Matt Graham'
__license__ = 'MIT'

import lasagne


class LasagneNetworkWrapper(object):

    def __init__(self, network, param_prec=0.):
        self.network = network
        self.param_prec = param_prec
        self.output_shape = network.output_shape

    @property
    def params(self):
        return lasagne.layers.get_all_params(self.network, trainable=True)

    def __call__(self, x):
        return lasagne.layers.get_output(self.network, x)

    def param_log_prior(self):
        log_prior = 0.
        for param in self.params:
            log_prior -= 0.5 * self.param_prec * (param**2).sum()
        return log_prior


class IdentityMap(object):

    @property
    def params(self):
        return []

    def __call__(self, x):
        return x

    def param_log_prior(self):
        return 0


class ParametricNonlinearityLayer(lasagne.layers.Layer):

    def __init__(self, incoming, alpha=lasagne.init.Constant(1.),
                 shared_axes='auto', **kwargs):
        super(ParametricNonlinearityLayer, self).__init__(incoming, **kwargs)
        if shared_axes == 'auto':
            self.shared_axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif shared_axes == 'all':
            self.shared_axes = tuple(range(len(self.input_shape)))
        elif isinstance(shared_axes, int):
            self.shared_axes = (shared_axes,)
        else:
            self.shared_axes = shared_axes

        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.shared_axes]
        if any(size is None for size in shape):
            raise ValueError("ParametricNonlinearityLayer needs input sizes "
                             "for all axes that alpha's are not shared over.")
        self.alpha = self.add_param(alpha, shape, name="alpha",
                                    regularizable=False)

    def get_output_for(self, input, **kwargs):
        axes = iter(range(self.alpha.ndim))
        pattern = ['x' if input_axis in self.shared_axes
                   else next(axes)
                   for input_axis in range(input.ndim)]
        alpha = self.alpha.dimshuffle(pattern)
        return 0.5 * (((input**2 + 1)**0.5 + input)**alpha -
                      ((input**2 + 1)**0.5 - input)**alpha)
