# -*- coding: utf-8 -*-
"""Model learning rules."""

__authors__ = 'Matt Graham'
__license__ = 'MIT'

import numpy as np
import theano as th
import theano.tensor as tt
import time
import logging

logger = logging.getLogger(__name__)


def floatx_array(val):
    """Helper for enforcing numpy array to be of theano.config.floatX dtype."""
    return np.array(val).astype(th.config.floatX)


class BaseLearningRule(object):

    def __init__(self):
        self.step_mult = th.shared(floatx_array(1.), name='step_mult')
        self.step_count = th.shared(0, name='step_count')
        self.step_updates = [(self.step_count, self.step_count + 1)]

    def mult_step_size(self, factor):
        self.step_mult.set_value(self.step_mult.get_value() * factor)

    def reset_state(self):
        self.step_count.set_value(0)

    def add_step_updates(self, params, obj_grads):
        raise NotImplementedError()

    def setup(self, params, obj_grads, dataset, x_batch):
        batch_index, batch_size = tt.iscalars('batch_index', 'batch_size')
        x_train_batch_sub = {
            x_batch: dataset.get_train_batch(batch_index, batch_size)}
        self.add_step_updates(params, obj_grads)
        start_time = time.time()
        logger.info('Compiling step function...')
        self.perform_step = th.function(
            inputs=[batch_index, batch_size],
            givens=x_train_batch_sub,
            updates=self.step_updates)
        logger.info('... finished in {0:.1f}s'
                    .format(time.time() - start_time))


class SGDWithMomentum(BaseLearningRule):
    """Stochastic gradient descent with momentum learning rule."""

    def __init__(self, step_size_sched, mom_coeff_sched):
        super(SGDWithMomentum, self).__init__()
        self.step_size_sched = step_size_sched
        self.mom_coeff_sched = mom_coeff_sched

    def reset_state(self):
        super(SGDWithMomentum, self).reset_state()
        if hasattr(self, 'momentums'):
            for mom in self.momentums:
                mom.set_value(np.zeros_like(mom.get_value()))

    def add_step_updates(self, params, obj_grads):
        # create momentum shared variable for each parameters
        # and initialise to zero
        self.momentums = [th.shared(param.get_value() * 0.,
                                    name='mom_' + param.name)
                          for param in params]
        step_size = self.step_size_sched(self.step_count) * self.step_mult
        mom_coeff = self.mom_coeff_sched(self.step_count)
        self.step_updates += [
            (mom, mom_coeff * mom - step_size * grad)
            for mom, grad in zip(self.momentums, obj_grads)
        ]
        self.step_updates += [
            (param, param + mom)
            for param, mom in
            zip(params, self.momentums)
        ]


class Adam(BaseLearningRule):
    """
    Adam learning rule.

    Stochastic gradient descent based optimiser using adaptive estimates of
    lower-order moments [1].

    References
    ----------

    [1] Kingma & Welling, Adam: a method for stochastic optimisation, ICLR 2015
    """

    def __init__(self, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super(Adam, self).__init__()
        self.alpha = th.shared(floatx_array(alpha), name='alpha')
        self.beta_1 = th.shared(floatx_array(beta_1), name='beta_1')
        self.beta_2 = th.shared(floatx_array(beta_2), name='beta_2')
        self.epsilon = th.shared(floatx_array(epsilon), name='epsilon')

    def reset_state(self):
        super(Adam, self).reset_state()
        if hasattr(self, 'grad_mom_1_ests'):
            for mom_1 in self.grad_mom_1_ests:
                mom_1.set_value(np.zeros_like(mom_1.get_value()))
        if hasattr(self, 'grad_mom_2_ests'):
            for mom_2 in self.grad_mom_2_ests:
                mom_2.set_value(np.zeros_like(mom_2.get_value()))

    def add_step_updates(self, params, obj_grads):
        self.grad_mom_1_ests = [th.shared(param.get_value() * 0.,
                                          name=param.name + '_grad_mom_1_est')
                                for param in params]
        self.grad_mom_2_ests = [th.shared(param.get_value() * 0.,
                                          name=param.name + '_grad_mom_2_est')
                                for param in params]
        float_step_count = tt.cast(self.step_count + 1, 'float32')
        self.step_updates += [
            (grad_mom_1_est,
             grad_mom_1_est * self.beta_1 + (1 - self.beta_1) * grad)
            for grad, grad_mom_1_est in zip(obj_grads, self.grad_mom_1_ests)]
        self.step_updates += [
            (grad_mom_2_est,
             grad_mom_2_est * self.beta_2 + (1 - self.beta_2) * grad**2)
            for grad, grad_mom_2_est in zip(obj_grads, self.grad_mom_2_ests)]
        alpha_t = (self.alpha * self.step_mult *
                   (1 - self.beta_2**float_step_count)**0.5 /
                   (1 - self.beta_1**float_step_count))
        self.step_updates += [
            (param, param - alpha_t *
             grad_mom_1_est / (grad_mom_2_est**0.5 + self.epsilon))
            for param, grad_mom_1_est, grad_mom_2_est in
            zip(params, self.grad_mom_1_ests, self.grad_mom_2_ests)]
