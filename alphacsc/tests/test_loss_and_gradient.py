import pytest
import numpy as np
from functools import partial
from scipy.optimize import approx_fprime

from alphacsc.utils import get_D, construct_X_multi
from alphacsc.utils.compute_constants import compute_DtD
from alphacsc.utils.whitening import whitening
from alphacsc.loss_and_gradient import (gradient_d, gradient_zi,
                                        compute_X_and_objective_multi,
                                        _l2_gradient_zi,
                                        _dense_transpose_convolve_d)


def _gradient_zi(X, z, D, loss, loss_params, flatten=False):
    # pre-computed 'constant' variables are NOT used
    return gradient_zi(X[0], z[0], D, loss=loss, flatten=flatten,
                       loss_params=loss_params)


def _construct_X(X, z, D, loss, loss_params):
    return construct_X_multi(z, D, n_channels=X.shape[1])


def _objective(X, z, D, loss, loss_params):
    # pre-computed 'constant' variables are NOT used
    return compute_X_and_objective_multi(X, z, D, feasible_evaluation=False,
                                         loss=loss, loss_params=loss_params)


def _gradient_d(X, z, D, loss, loss_params, flatten=False):
    # pre-computed 'constant' variables are NOT used
    return gradient_d(D, X, z, loss=loss, flatten=flatten,
                      loss_params=loss_params)


def gradient_checker(func, grad, shape, args=(), kwargs={}, n_checks=10,
                     rtol=1e-5, grad_name='gradient', debug=False,
                     random_seed=None):
    """Check that the gradient correctly approximate the finite difference
    """

    rng = np.random.RandomState(random_seed)

    msg = ("Computed {} did not match gradient computed with finite "
           "difference. Relative error is {{}}".format(grad_name))

    func = partial(func, **kwargs)

    def test_grad(z0):
        grad_approx = approx_fprime(z0, func, 1e-8, *args)
        grad_compute = grad(z0, *args, **kwargs)
        error = np.sum((grad_approx - grad_compute) ** 2)
        error /= np.sum(grad_approx ** 2)
        error = np.sqrt(error)
        try:
            assert error < rtol, msg.format(error)
        except AssertionError:
            if debug:
                import matplotlib.pyplot as plt
                plt.plot(grad_approx, label="grad. approx.")
                plt.plot(grad_compute, label="grad. computed")
                plt.title("{}".format(grad_name))
                plt.legend()
                plt.grid()
                plt.show()
            raise

    z0 = np.zeros(shape)
    test_grad(z0)

    for _ in range(n_checks):
        z0 = rng.randn(shape)
        test_grad(z0)


def _set_up(seed=None):
    """ set-up the test
    """
    rng = np.random.RandomState(seed)

    n_trials, n_channels, n_times = 5, 3, 100
    n_atoms, n_times_atom = 10, 15

    n_times_valid = n_times - n_times_atom + 1

    X = rng.randn(n_trials, n_channels, n_times)
    z = rng.randn(n_trials, n_atoms, n_times_valid)

    uv = rng.randn(n_atoms, n_channels + n_times_atom)
    D = get_D(uv, n_channels)

    return X, uv, D, z, n_times_valid, n_atoms, n_channels


@pytest.mark.parametrize('loss', ['l2', 'l2_tv', 'dtw', 'whitening'])
@pytest.mark.parametrize('func', [
    _construct_X, _gradient_zi, _objective, _gradient_d])
def test_consistency(loss, func):
    """Check that the result are the same for the full rank D and rank 1 uv.
    """
    X, uv, D, z, _, _, _ = _set_up()

    loss_params = dict(gamma=.01)

    if loss == "whitening":
        loss_params['ar_model'], X = whitening(X)

    loss_params['block'] = True if (loss == "l2_tv") else False
    if loss_params['block']:  # set params for special case 'l2-tv'
        loss = "l2"

    val_D = func(X, z, D, loss, loss_params=loss_params)
    val_uv = func(X, z, uv, loss, loss_params=loss_params)

    assert np.allclose(val_D, val_uv)


@pytest.mark.parametrize('loss', ['l2', 'l2_tv', 'dtw', 'whitening'])
def test_gradients(loss):
    """Check that the gradients with approx_fprime Scipy function.
    """
    X, _, D, z, n_times_valid, n_atoms, n_channels = _set_up()

    loss_params = dict(gamma=.01)

    if loss == "whitening":
        loss_params['ar_model'], X = whitening(X)

    loss_name = loss  # for error message
    n_checks = 1 if loss == "dtw" else 5

    loss_params['block'] = True if (loss == "l2_tv") else False
    if loss_params['block']:  # set params for special case 'l2_tv'
        loss = "l2"
        loss_name = "l2_tv"

    # Test gradient D
    # pre-computed 'constant' variables are NOT used
    assert D.shape == _gradient_d(X, z, D, loss, loss_params=loss_params).shape

    def pobj_d(ds):
        return _objective(X, z, ds.reshape(n_atoms, n_channels, -1), loss,
                          loss_params=loss_params)

    def grad_d(ds):
        return _gradient_d(X, z, ds, loss=loss, flatten=True,
                           loss_params=loss_params)

    _grad_name = "gradient D for loss '{}'".format(loss_name)

    gradient_checker(pobj_d, grad_d, np.prod(D.shape), n_checks=n_checks,
                     debug=True, grad_name=_grad_name, rtol=1e-4)

    # Test gradient z
    # pre-computed 'constant' variables are NOT used
    assert z[0].shape == _gradient_zi(
        X, z, D, loss, loss_params=loss_params).shape

    def pobj_z(zs):
        return _objective(X[:1], zs.reshape(1, n_atoms, -1), D, loss,
                          loss_params=loss_params)

    def grad_z(zs):
        return gradient_zi(X[0], zs, D, loss=loss,
                           flatten=True, loss_params=loss_params)

    _grad_name = "gradient z for loss '{}'".format(loss_name)

    gradient_checker(pobj_z, grad_z, n_atoms * n_times_valid,
                     n_checks=n_checks, debug=True, grad_name=_grad_name,
                     rtol=1e-4)


@pytest.mark.parametrize('ds', ['D', 'uv'])
@pytest.mark.parametrize('loss_params', [None, dict(block=True)])
def test__l2_gradient_zi_consistency(ds, loss_params):
    """Check that the _l2_gradient_zi provide the same output with and without
    constant (pre-computed) variables
    """
    X, uv, D, z, _, _, n_channels = _set_up(0)

    if ds == 'D':
        DtD = compute_DtD(D, n_channels=n_channels)
        DtX_i = _dense_transpose_convolve_d(X[0], D=D, n_channels=n_channels)
        constants = dict(DtD=DtD, DtX_i=DtX_i)

    if ds == 'uv':
        DtD = compute_DtD(uv, n_channels=n_channels)
        DtX_i = _dense_transpose_convolve_d(X[0], D=uv, n_channels=n_channels)
        constants = dict(DtD=DtD, DtX_i=DtX_i)

    _, grad_with_cst = _l2_gradient_zi(X[0], z[0], D=D,
                                       constants=constants,
                                       loss_params=dict(block=True),
                                       return_func=False)

    _, grad_without_cst = _l2_gradient_zi(X[0], z[0], D=D,
                                          constants=None,
                                          loss_params=dict(block=True),
                                          return_func=False)

    np.testing.assert_allclose(grad_with_cst, grad_without_cst)


@pytest.mark.parametrize('loss', ['l2', 'l2_tv', 'dtw', 'whitening'])
def test_loss_consistency(loss):
    """Check that the loss are consistent w.r.t the gradient_zi computation and
    the dedicated function
    """
    rng = np.random.RandomState(None)

    n_trials, n_channels, n_times = 1, 3, 100
    n_atoms, n_times_atom = 10, 15
    n_times_valid = n_times - n_times_atom + 1

    X = rng.randn(n_trials, n_channels, n_times)
    z = rng.randn(n_trials, n_atoms, n_times_valid)

    D = rng.randn(n_atoms, n_channels, n_times_atom)
    reg = 1.0

    loss_params = dict(gamma=.01)

    if loss == "whitening":
        loss_params['ar_model'], X = whitening(X)

    loss_params['block'] = True if (loss == "l2_tv") else False
    if loss_params['block']:  # set params for special case 'l2_tv'
        loss = "l2"

    cost_ref = compute_X_and_objective_multi(
                                X, z, D_hat=D, reg=reg, loss=loss,
                                loss_params=loss_params,
                                feasible_evaluation=False,
                                uv_constraint='separate', return_X_hat=False
                                )
    cost_test, _ = gradient_zi(X[0], z[0], D=D, constants=None, reg=reg,
                               loss=loss, loss_params=loss_params,
                               return_func=True, flatten=False)

    np.testing.assert_allclose(cost_ref, cost_test)
