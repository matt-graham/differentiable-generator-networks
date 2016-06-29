# -*- coding: utf-8 -*-
"""Classes for managing data to be modelled."""

__authors__ = 'Matt Graham'
__license__ = 'MIT'

import numpy as np
import theano as th
import logging
import utils

logger = logging.getLogger(__name__)


class Dataset(object):
    """ Basic dataset class. """

    def __init__(self, data, n_valid, corruptor=None, prng=None):
        """
        Parameters
        ----------
        data : numpy array
            Data matrix array with rows corresponding to data vectors.
        n_valid : integer
            Number of data vectors to use as validation set.
        corruptor : function(Array, RandomState) or None
            Optional function which applies random 'corruption' / augmentation
            to data, for example dequantising pixel values, adding noise,
            applying random affine transformation to image. Applied on
            initialisation and at end of each training epoch.
        prng : RandomState or None
            Seeded pseudo-random number generator - used to shuffle data
            and for corruptor if specified.
        """
        self.data = data
        self.n_valid = n_valid
        self.n_train = data.shape[0] - n_valid
        self.corruptor = corruptor
        if prng is None:
            prng = np.random.RandomState()
        self.prng = prng
        shuffled_data, self.perm = utils.shuffle(self.data, self.prng)
        self.data_valid, self.data_train = utils.split(shuffled_data, n_valid)
        if corruptor is None:
            self.x_valid = th.shared(
                self.data_valid.astype(th.config.floatX), 'x_valid')
            self.x_train = th.shared(
                self.data_train.astype(th.config.floatX), 'x_train')
        else:
            corrupted_data_valid = self.corruptor(self.data_valid, self.prng)
            corrupted_data_train = self.corruptor(self.data_train, self.prng)
            self.x_valid = th.shared(
                corrupted_data_valid.astype(th.config.floatX), 'x_valid')
            self.x_train = th.shared(
                corrupted_data_train.astype(th.config.floatX), 'x_train')

    def get_train_batch(self, batch_index, batch_size):
        return self.x_train[batch_index * batch_size:
                            (batch_index + 1) * batch_size]

    def end_of_epoch_callback(self):
        if self.corruptor is not None:
            data_train = self.corruptor(self.data_train, self.prng)
            self.x_train.set_value(data_train.astype(th.config.floatX))


class AuxilliaryVariableDataset(Dataset):
    """ Dataset class for models with inputs formed of data vector plus
        independent auxillary random vector."""

    def __init__(self, data, n_valid, auxilliary_sampler,
                 data_corruptor=None, prng=None):

        def corruptor(data, prng):
            if data_corruptor is not None:
                data = data_corruptor(data, prng)
            n_data = data.shape[0]
            aux_vars = auxilliary_sampler(n_data, prng)
            return np.hstack((data, aux_vars))

        super(AuxilliaryVariableDataset, self).__init__(
            data, n_valid, corruptor, prng)
