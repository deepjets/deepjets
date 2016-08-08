from nose.tools import (raises, assert_raises, assert_true,
                        assert_equal, assert_not_equal, assert_almost_equal)

import numpy as np

from deepjets.utils import default_inv_roc_curve, lklhd_inv_roc_curve, lklhd_inv_roc_curve2d
from sklearn.metrics import auc
from scipy import interpolate


def test_default_vs_likelihood():

    sig, bkd = np.random.normal(2, 1, 1000000), np.random.normal(0, 1, 1000000)
    y_true = np.concatenate([np.repeat([[1, 0]], sig.shape[0], axis=0),
                             np.repeat([[0, 1]], bkd.shape[0], axis=0)])
    y_pred = np.concatenate([sig, bkd])

    roc_def = default_inv_roc_curve(y_true, y_pred)
    roc_lkl = lklhd_inv_roc_curve(y_true, y_pred, nb_per_bin=100)

    assert_true(
        abs(interpolate.interp1d(roc_def[:,0], roc_def[:,1])(0.5) -
            interpolate.interp1d(roc_lkl[:,0], roc_lkl[:,1])(0.5)) < 3)


def test_1d_vs_2d_likelihood():
    sig, bkd = np.random.normal(2, 1, 1000000), np.random.normal(0, 1, 1000000)
    y_true = np.concatenate([np.repeat([[1, 0]], sig.shape[0], axis=0),
                             np.repeat([[0, 1]], bkd.shape[0], axis=0)])
    y_pred = np.concatenate([sig, bkd])

    roc_lkl = lklhd_inv_roc_curve(y_true, y_pred, nb_per_bin=100)
    roc_lkl_2d = lklhd_inv_roc_curve2d(y_true, y_pred, y_pred, nb_per_bin=100)

    assert_true(
        abs(interpolate.interp1d(roc_lkl[:,0], roc_lkl[:,1])(0.5) -
            interpolate.interp1d(roc_lkl_2d[:,0], roc_lkl_2d[:,1])(0.5)) < 3)
