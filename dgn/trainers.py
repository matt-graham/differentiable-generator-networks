# -*- coding: utf-8 -*-
"""Optimisation-based trainers for differentiable generator network models."""

__authors__ = 'Matt Graham'
__license__ = 'MIT'

import os
import datetime
import time
import numpy as np
import theano as th
import theano.tensor as tt
import logging


logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, model, dataset, learning_rule, prng,
                 batch_size=100, save_current_interval=-1,
                 save_best=True, save_dir=None, tag='',
                 do_nan_update_check=True,
                 invalid_update_ss_mult_factor=0.5,
                 end_of_epoch_callback=None):
        self.model = model
        self.dataset = dataset
        self.learning_rule = learning_rule
        self.prng = prng
        self.batch_size = batch_size
        self.save_current_interval = save_current_interval
        self.save_best = save_best
        self.save_dir = save_dir if save_dir is not None else os.getcwd()
        self.tag = tag
        self.do_nan_update_check = do_nan_update_check
        self.invalid_update_ss_mult_factor = invalid_update_ss_mult_factor
        self.end_of_epoch_callback = end_of_epoch_callback
        self._compile_theano_functions()
        self.params_best = self.model.get_current_params_dict()
        self.best_obj_valid = self.objective_valid()

    def _compile_theano_functions(self):
        x_batch = tt.matrix('x_batch')
        obj = self.model.objective(x_batch)
        obj_grads = [tt.grad(obj, param) for param in self.model.params]
        self.learning_rule.setup(self.model.params, obj_grads,
                                 self.dataset, x_batch)
        self.objective_train = th.function(
            [], obj, givens={x_batch: self.dataset.x_train})
        self.objective_valid = th.function(
            [], obj, givens={x_batch: self.dataset.x_valid})

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value
        self._n_batches = self.dataset.n_train // self._batch_size
        if self.dataset.n_train % self._batch_size != 0:
            logger.warn('Training set size not an integer multiple of batch '
                        'size. Some training examples will be ignored.')

    @property
    def n_batches(self):
        return self._n_batches

    def restore_best_params(self):
        if hasattr(self, 'params_best'):
            for param in self.model.params:
                param.set_value(self.params_best[param.name])
        else:
            raise Exception('Trying to restore best parameters but no '
                            'training performed yet.')

    def train(self, n_epoch=100, overwrite_previous=True):
        logger.info('-' * 40)
        logger.info('Starting training, current best obj(valid): {0}'
                    .format(self.best_obj_valid))
        if not hasattr(self, 'file_prefix') or overwrite_previous:
            tstamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_')
            self.file_prefix = self.tag + '_' + tstamp if self.tag else tstamp
            logger.info('Parameters saved to {0}/{1}params_*.npz'
                        .format(self.save_dir, self.file_prefix))
        logger.info('-' * 40)
        for e in range(n_epoch):
            batch_order = self.prng.permutation(self.n_batches)
            start_time = time.time()
            for batch_index in batch_order:
                    self.learning_rule.perform_step(
                        batch_index, self.batch_size)
            obj_train = self.objective_train()
            obj_valid = self.objective_valid()
            epoch_time = time.time() - start_time
            logger.info('Epoch {0}: obj(train)={1:.2f} obj(valid)={2:.2f}, '
                        'time={3:.1f}s'.format(e + 1, float(obj_train),
                                               float(obj_valid), epoch_time))
            self.dataset.end_of_epoch_callback()
            if self.end_of_epoch_callback is not None:
                self.end_of_epoch_callback(self)
            if self.do_nan_update_check:
                self.nan_update_check(obj_train, obj_valid)
            if obj_valid < self.best_obj_valid:
                self.params_best = self.model.get_current_params_dict()
                self.best_obj_valid = obj_valid
                if self.save_best:
                    file_name = '{0}_params_best.npz'.format(self.file_prefix)
                    file_path = os.path.join(self.save_dir, file_name)
                    logger.info('Saving new best parameters.')
                    self.model.save_params(file_path, self.params_best)
            if (self.save_current_interval > 0 and
                    e % self.save_current_interval == 0):
                file_name = '{0}_params_current.npz'.format(self.file_prefix)
                file_path = os.path.join(self.save_dir, file_name)
                logger.info('Saving current parameters.')
                self.model.save_params(file_path)
        logger.info('-' * 40)
        logger.info('Finished training, new best obj(valid): {0}'
                    .format(self.best_obj_valid))

    def nan_update_check(self, obj_train, obj_valid):
        nan_update = any([np.any(np.isnan(param.get_value()))
                          for param in self.model.params])
        if (nan_update or np.isinf(obj_train) or np.isnan(obj_train) or
                np.isinf(obj_valid) or np.isnan(obj_valid)):
            self.restore_best_params()
            obj_valid = self.best_obj_valid
            self.learning_rule.reset_state()
            self.learning_rule.mult_step_size(
                self.invalid_update_ss_mult_factor)
            logger.info('Update set some parameters to NaN or gave NaN/Inf'
                        ' training or validation set objective. '
                        'Restoring previous best parameters values, '
                        'resetting learning rule state and reducing step '
                        'size.')
