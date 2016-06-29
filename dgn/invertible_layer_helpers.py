# -*- coding: utf-8 -*-
"""Invertible layer helper functions."""

__authors__ = 'Matt Graham'
__license__ = 'MIT'

import numpy as np
import theano.tensor as tt
import dgn.invertible_layers as layers


def alt_lower_upper_tri_layers(n_layer, weights_inits, biases_inits,
                               nl_fwd=tt.sinh, nl_inv=tt.arcsinh,
                               weights_prec=0., biases_prec=0.):
    layers = []
    for l in range(n_layer):
        if l % 2 == 0:
            layers.append(TriangularAffineLayer(
                weights_init=np.tril(weights_inits[l]),
                biases_init=biases_inits[l],
                lower=True,
                weights_prec=weights_prec,
                biases_prec=biases_prec))
            layers.append(ElementwiseLayer(nl_fwd, nl_inv))
        else:
            layers.append(TriangularAffineLayer(
                weights_init=np.triu(weights_inits[l]),
                biases_init=biases_inits[l],
                lower=False,
                weights_prec=weights_prec,
                biases_prec=biases_prec))
            layers.append(ElementwiseLayer(nl_fwd, nl_inv))
    return layers


def alt_lower_upper_tri_with_fwd_diag_inv_nl_layers(
        n_layer, weights_inits, biases_inits, diag_weights_inits,
        nl_fwd=tt.sinh, nl_inv=tt.arcsinh, weights_prec=0., biases_prec=0.,
        diag_weights_prec=0.):
    layers = []
    for l in range(n_layer):
        if l % 2 == 0:
            layers.append(TriangularAffineLayer(
                weights_init=np.tril(weights_inits[l]),
                biases_init=biases_inits[2*l],
                lower=True,
                weights_prec=weights_prec,
                biases_prec=biases_prec))
            layers.append(FwdDiagInvElementwiseLayer(
                forward_func=nl_fwd,
                inverse_func=nl_inv,
                diag_weights_init=diag_weights_inits[l],
                biases_init=biases_inits[2 * l + 1],
                diag_weights_prec=weights_prec,
                biases_prec=biases_prec))
        else:
            layers.append(TriangularAffineLayer(
                weights_init=np.triu(weights_inits[l]),
                biases_init=biases_inits[l],
                lower=False,
                weights_prec=weights_prec,
                biases_prec=biases_prec))
            layers.append(FwdDiagInvElementwiseLayer(
                forward_func=nl_fwd,
                inverse_func=nl_inv,
                diag_weights_init=diag_weights_inits[l],
                biases_init=biases_inits[2 * l + 1],
                diag_weights_prec=diag_weights_prec,
                biases_prec=biases_prec))
    return layers


def diag_plus_rank_1_with_fwd_diag_inv_nl_layers(
        n_layer, diag_weights_inits, u_vect_inits, v_vect_inits, biases_inits,
        nl_fwd=tt.sinh, nl_inv=tt.arcsinh, diag_weights_prec=0.,
        u_vect_prec=0., v_vect_prec=0., biases_prec=0.):
    layers = []
    for l in range(n_layer):
        layers.append(DiagPlusRank1AffineLayer(
            diag_weights_init=diag_weights_inits[2 * l],
            u_vect_init=u_vect_inits[l],
            v_vect_init=v_vect_inits[l],
            biases_init=biases_inits[2 * l],
            diag_weights_prec=diag_weights_prec,
            u_vect_prec=u_vect_prec,
            v_vect_prec=v_vect_prec,
            biases_prec=biases_prec))
        layers.append(FwdDiagInvElementwiseLayer(
            forward_func=nl_fwd,
            inverse_func=nl_inv,
            diag_weights_init=diag_weights_inits[2 * l + 1],
            biases_init=biases_inits[2 * l + 1],
            diag_weights_prec=diag_weights_prec,
            biases_prec=biases_prec))
    return layers
