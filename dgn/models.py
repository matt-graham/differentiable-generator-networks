# -*- coding: utf-8 -*-
"""
Differentiable generator network models.
"""

__authors__ = 'Matt Graham'
__license__ = 'MIT'

import theano as th
import theano.tensor as tt
import numpy as np
import datetime
import logging
import os
import time
from theano.compile.nanguardmode import NanGuardMode

logger = logging.getLogger(__name__)


class BaseModel(object):

    def __init__(self, params):
        # make sure variables all have unique names as used for dict keys
        th.gof.utils.give_variables_names(params)
        self.params = params

    def objective(self, x_batch):
        raise NotImplementedError()

    def get_current_params_dict(self):
        return {param.name: param.get_value(borrow=False)
                for param in self.params}

    def save_params(self, file_path, params_dict=None):
        if params_dict is None:
            params_dict = {param.name: param.get_value()
                           for param in self.params}
        np.savez(file_path, **params_dict)

    def load_params(self, file_path):
        loaded_params = np.load(file_path)
        # check all parameters present in file and valid shape/dtype
        # before changing any values
        for param in self.params:
            assert param.name in loaded_params.keys(), (
                'Parameter file does contain parameter {0}.'
                .format(param.name))
            val = param.get_value()
            assert val.shape == loaded_params[param.name].shape, (
                'Shape of parameter {0} in parameter file is invalid.'
                .format(param.name))
            assert val.dtype == loaded_params[param.name].dtype, (
                'Dtype of parameter {0} in parameter file is invalid.'
                .format(param.name))
        # now change to parameter values loaded from file
        for param in self.params:
            param.set_value(loaded_params[param.name])


class BaseVariationalAutoencoder(BaseModel):

    def __init__(self, encoder, decoder, random_stream,
                 param_prior_prec=0.):
        params = self._extract_params_and_rename(encoder, decoder)
        super(BaseVariationalAutoencoder, self).__init__(params)
        self.encoder = encoder
        self.decoder = decoder
        self.z_dim = encoder.output_shape[-1] // 2
        self.random_stream = random_stream
        self.param_prior_prec = param_prior_prec
        self._compile_theano_functions()

    def z_gvn_x(self, x_batch):
        mean_log_var = self.encoder(x_batch)
        return mean_log_var[:, :self.z_dim], mean_log_var[:, self.z_dim:]

    def x_gvn_z(self, z_batch):
        raise NotImplementedError()

    def sample_z_gvn_x(self, x_batch):
        mean_z_gvn_x, log_var_z_gvn_x = self.z_gvn_x(x_batch)
        ns = self.random_stream.normal(mean_z_gvn_x.shape)
        return mean_z_gvn_x + np.exp(0.5 * log_var_z_gvn_x) * ns

    def generator(self, u):
        raise NotImplementedError()

    def _extract_params_and_rename(self, encoder, decoder):
        for param in encoder.params:
            if param.name[:4] != 'enc_':
                param.name = 'enc_' + param.name
        for param in decoder.params:
            if param.name[:4] != 'dec_':
                param.name = 'dec_' + param.name
        return encoder.params + decoder.params

    def sample_inputs(self, n_sample):
        raise NotImplementedError()

    def _compile_theano_functions(self):
        n_sample = tt.iscalar('n_sample')
        us = self.sample_inputs(n_sample)
        xs = self.generator(us)
        self.generate = th.function([n_sample], xs)

    def kl_div_q_prior(self, x_batch):
        mean_z_gvn_x, log_var_z_gvn_x = self.z_gvn_x(x_batch)
        return -0.5 * (1 + log_var_z_gvn_x - mean_z_gvn_x**2 -
                       tt.exp(log_var_z_gvn_x)).sum()

    def expc_q_log_p_est(self, x_batch, z_samples):
        raise NotImplementedError()

    def param_neg_log_prior(self):
        neg_log_prior = 0.
        for param in self.params:
            neg_log_prior += 0.5 * (param**2).sum()
        return neg_log_prior * self.param_prior_prec

    def objective(self, x_batch):
        kl_q_prior = self.kl_div_q_prior(x_batch)
        z_samples = self.sample_z_gvn_x(x_batch)
        expc_q_log_p_est = self.expc_q_log_p_est(x_batch, z_samples)
        batch_size = tt.cast(x_batch.shape[0], th.config.floatX)
        obj = (kl_q_prior - expc_q_log_p_est) / batch_size
        if self.param_prior_prec != 0.:
            obj = obj + self.param_neg_log_prior()
        return obj


class GaussianVariationalAutoencoder(BaseVariationalAutoencoder):

    def __init__(self, encoder, decoder, random_stream,
                 param_prior_prec=0.):
        self.x_dim = decoder.output_shape[-1] // 2
        super(GaussianVariationalAutoencoder, self).__init__(
            encoder, decoder, random_stream, param_prior_prec
        )

    def x_gvn_z(self, z_batch):
        mean_log_var = self.decoder(z_batch)
        return mean_log_var[:, :self.x_dim], mean_log_var[:, self.x_dim:]

    def generator(self, u):
        if u.ndim == 1:
            u = u[None, :]
        z, n = u[:, :self.z_dim], u[:, self.z_dim:self.z_dim + self.x_dim]
        mean, log_var = self.x_gvn_z(z)
        return tt.squeeze(mean + tt.exp(0.5 * log_var) * n)

    def sample_inputs(self, n_sample):
        return self.random_stream.normal((n_sample, self.z_dim + self.x_dim))

    def expc_q_log_p_est(self, x_batch, z_samples):
        mean_x_gvn_z, log_var_x_gvn_z = self.x_gvn_z(z_samples)
        return -0.5 * (
            (x_batch - mean_x_gvn_z)**2 / tt.exp(log_var_x_gvn_z) +
            log_var_x_gvn_z).sum()


class BernoulliVariationalAutoencoder(BaseVariationalAutoencoder):

    def __init__(self, encoder, decoder, random_stream,
                 param_prior_prec=0.):
        self.x_dim = decoder.output_shape[-1]
        super(BernoulliVariationalAutoencoder, self).__init__(
            encoder, decoder, random_stream, param_prior_prec
        )

    def x_gvn_z(self, z_batch):
        return self.decoder(z_batch)

    def generator(self, u):
        if u.ndim == 1:
            u = u[None, :]
        z, n = u[:, :self.z_dim], u[:, self.z_dim:self.z_dim + self.x_dim]
        prob_on = self.x_gvn_z(z)
        return (n < prob_on) * 1.

    def sample_inputs(self, n_sample):
        zs = self.random_stream.normal((n_sample, self.z_dim))
        ns = self.random_stream.uniform((n_sample, self.x_dim))
        return tt.concatenate([zs, ns], 1)

    def expc_q_log_p_est(self, x_batch, z_samples):
        prob_on = self.x_gvn_z(z_samples)
        return (x_batch * tt.log(prob_on) +
                (1. - x_batch) * tt.log(1. - prob_on)).sum()


class WrappedCauchyVariationalAutoencoder(BaseVariationalAutoencoder):

    def __init__(self, encoder, decoder, random_stream,
                 param_prior_prec=0.):
        self.x_dim = decoder.output_shape[-1] // 2
        super(WrappedCauchyVariationalAutoencoder, self).__init__(
            encoder, decoder, random_stream, param_prior_prec
        )

    def x_gvn_z(self, z_batch):
        loc_scale = self.decoder(z_batch)
        return 2 * np.pi * loc_scale[:, :self.x_dim], loc_scale[:, self.x_dim:]

    def generator(self, u):
        if u.ndim == 1:
            u = u[None, :]
        z, n = u[:, :self.z_dim], u[:, self.z_dim:self.z_dim + self.x_dim]
        loc, scale = self.x_gvn_z(z)
        alpha = 2 * scale / (1 + scale**2)
        phi = tt.arccos((tt.cos(2 * np.pi * n) + alpha) /
                        (1 + alpha * tt.cos(2 * np.pi * n))) + loc
        return tt.switch(n < 0.5, phi, 2 * np.pi - phi)

    def sample_inputs(self, n_sample):
        zs = self.random_stream.normal((n_sample, self.z_dim))
        ns = self.random_stream.uniform((n_sample, self.x_dim))
        return tt.concatenate([zs, ns], 1)

    def expc_q_log_p_est(self, x_batch, z_samples):
        loc, scale = self.x_gvn_z(z_samples)
        return (tt.log(1. - scale**2) - tt.log(2 * np.pi) -
                tt.log(1 + scale**2 - 2 * scale * tt.cos(x_batch - loc))).sum()


class DeepInvertibleGenerator(BaseModel):

    def __init__(self, layers, base_log_density,
                 compile_nll_x_grad=False,
                 compile_jacobian_funcs=False,
                 debug_mode=False):
        params = self._extract_params_and_rename(layers)
        super(DeepInvertibleGenerator, self).__init__(params)
        self.layers = layers
        self.base_log_density = base_log_density
        self.compile_nll_x_grad = compile_nll_x_grad
        self.compile_jacobian_funcs = compile_jacobian_funcs
        self.debug_mode = debug_mode
        if debug_mode:
            self.func_mode = NanGuardMode(nan_is_error=True, inf_is_error=True,
                                          big_is_error=True)
        else:
            self.func_mode = None
        logger.info('=' * 40)
        self._extract_params_and_rename()
        start_time = time.time()
        logger.info('Compiling theano functions...')
        self._compile_theano_functions()
        logger.info('... finished in {0:.1f}s'
                    .format(time.time() - start_time))
        logger.info('=' * 40)

    def _extract_params_and_rename(self, layers):
        params = []
        for l, layer in enumerate(layers):
            # add layer index to parameter names
            for param in layer.params:
                param.name = '{0}_{1}'.format(param.name, l)
            params += layer.params
        return params

    def symbolic_forward_map_and_log_likelihood(self, x):
        log_lik = 0.
        y = x
        for l, layer in enumerate(self.layers):
            log_lik += layer.forward_jacobian_log_det(y)
            if self.debug_mode:
                log_lik = th.printing.Print(
                    'Accumulated log-Jacobian determinants up to layer {0}.'
                    .format(l))(log_lik)
            y = layer.forward_map(y)
        log_lik += self.base_log_density(y)
        if self.debug_mode:
            log_lik = th.printing.Print(
                'Log-likelihood after adding base density terms.')(log_lik)
        return y, log_lik

    def symbolic_param_log_prior(self):
        log_prior = 0.
        for layer in self.layers:
            log_prior += layer.param_log_prior()
        return log_prior

    def objective(self, x_batch):
        y, log_lik = self.symbolic_forward_map_and_log_likelihood(x_batch)
        log_prior = self.symbolic_param_log_prior()
        batch_size = tt.cast(x_batch.shape[0], th.config.floatX)
        return -(log_lik + log_prior) / batch_size

    def symbolic_inverse_map(self, y):
        x = y
        for layer in self.layers[::-1]:
            x = layer.inverse_map(x)
        return x

    def _compile_theano_functions(self):
        x_batch = tt.matrix('x_batch')
        x_point = tt.vector('x_point')
        y_batch, log_lik_batch = (
            self.symbolic_forward_map_and_log_likelihood(x_batch))
        y_point, log_lik_point = (
            self.symbolic_forward_map_and_log_likelihood(x_point))
        log_prior = self.symbolic_param_log_prior()
        start_time = time.time()
        logger.info('  Compiling forward map functions...')
        self.forward_map_batch = th.function(
            inputs=[x_batch], outputs=y_batch, mode=self.func_mode)
        self.forward_map_point = th.function(
            inputs=[x_point], outputs=y_point, mode=self.func_mode)
        logger.info('  ... finished in {0:.1f}s'
                    .format(time.time() - start_time))
        mean_log_lik_batch = (log_lik_batch /
                              tt.cast(x_batch.shape[0], 'float32'))
        start_time = time.time()
        logger.info('  Compiling log likelihood functions...')
        self.log_lik_batch = th.function(
            inputs=[x_batch], outputs=log_lik_batch, mode=self.func_mode)
        self.log_lik_point = th.function(
            inputs=[x_point], outputs=log_lik_point, mode=self.func_mode)
        self.mean_log_lik_train = th.function(
            inputs=[], outputs=mean_log_lik_batch,
            givens={x_batch: self.dataset.x_train}, mode=self.func_mode)
        self.mean_log_lik_valid = th.function(
            inputs=[], outputs=mean_log_lik_batch,
            givens={x_batch: self.dataset.x_valid}, mode=self.func_mode)
        logger.info('  ... finished in {0:.1f}s'
                    .format(time.time() - start_time))
        if self.compile_nll_x_grad:
            log_lik_grad = tt.grad(log_lik_point, x_point)
            start_time = time.time()
            logger.info('  Compiling NLL gradient wrt x function...')
            self.neg_log_lik_grad = th.function(
                inputs=[x_point], outputs=-log_lik_grad)
            logger.info('  ... finshed in {0:.1f}s'
                        .format(time.time() - start_time))
        y_batch = tt.matrix('y_batch')
        y_point = tt.vector('y_point')
        start_time = time.time()
        logger.info('  Compiling inverse map functions...')
        self.inverse_map_batch = th.function(
            inputs=[y_batch], outputs=self.symbolic_inverse_map(y_batch),
            mode=self.func_mode)
        self.inverse_map_point = th.function(
            inputs=[y_point], outputs=self.symbolic_inverse_map(y_point),
            mode=self.func_mode)
        logger.info('  ... finished in {0:.1f}s'
                    .format(time.time() - start_time))
        if self.compile_jacobian_funcs:
            start_time = time.time()
            logger.info('  Compiling Jacobian functions...')
            y_rep = tt.tile(y_point, (y_point.shape[0], 1))
            x_rep = self.symbolic_inverse_map(y_rep)
            dx_dy = tt.grad(cost=None, wrt=y_rep,
                            known_grads={x_rep: tt.identity_like(x_rep)})
            dx_dy_prod_chol = tt.slinalg.cholesky(dx_dy.dot(dx_dy.T))
            self.inverse_map_jacobian = th.function(
                inputs=[y_point], outputs=dx_dy)
            self.inverse_map_jacobian_and_prod_chol = th.function(
                inputs=[y_point], outputs=[dx_dy, dx_dy_prod_chol])
            logger.info('  ... finished in {0:.1f}s'
                        .format(time.time() - start_time))

    def add_layers(self, extra_layers):
        for l, layer in enumerate(extra_layers):
            # add layer index to parameter names
            for param in layer.params:
                param.name = '{0}_{1}'.format(param.name, l + len(self.layers))
            self.params += layer.params
        # make sure variables all have unique names as
        # used for dict keys
        th.gof.utils.give_variables_names(self.params)
        self.layers += extra_layers
        start_time = time.time()
        logger.info('Recompiling theano functions...')
        self._compile_theano_functions()
        logger.info('... finished in {0:.1f}s'
                    .format(time.time() - start_time))
