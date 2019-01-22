import pytest
import numpy as np


from alphacsc.utils import check_random_state, get_D
from alphacsc.utils.whitening import whitening, apply_whitening
from alphacsc.utils.compute_constants import compute_DtD, compute_ztz
from alphacsc.utils.convolution import (tensordot_convolve, construct_X_multi,
                                        _choose_convolve_multi)
from alphacsc.loss_and_gradient import (_compute_DtD_z_i,
                                        _dense_transpose_convolve_d)


def test_DtD_z_i_computation():
    """ Test the computation of _compute_DtD_z_i
    """
    rng = np.random.RandomState(None)

    n_trials, n_channels, n_times = 1, 3, 100
    n_atoms, n_times_atom = 10, 15
    n_times_valid = n_times - n_times_atom + 1

    z = rng.randn(n_trials, n_atoms, n_times_valid)

    D = rng.randn(n_atoms, n_channels, n_times_atom)
    DtD = compute_DtD(D)

    D_z_i = _choose_convolve_multi(z[0], D=D, n_channels=n_channels)
    DtD_z_i_ = _dense_tr_conv_d(D_z_i, D=D, n_channels=n_channels)
    DtD_z_i = _compute_DtD_z_i(z[0], DtD=DtD)

    np.testing.assert_allclose(DtD_z_i_, DtD_z_i)


def test_DtD_consistency():
    """ Test the consistency of the DtD computation from uv and D
    """
    n_atoms = 10
    n_channels = 5
    n_times_atom = 50
    random_state = 42

    rng = check_random_state(random_state)

    uv = rng.randn(n_atoms, n_channels + n_times_atom)
    D = get_D(uv, n_channels)

    assert np.allclose(compute_DtD(uv, n_channels=n_channels),
                       compute_DtD(D))


@pytest.mark.parametrize('use_whitening', [False, True])
def test_ztz(use_whitening):
    n_atoms = 7
    n_trials = 3
    n_channels = 5
    n_times_valid = 500
    n_times_atom = 10
    n_times = n_times_valid + n_times_atom - 1
    random_state = None

    rng = check_random_state(random_state)

    X = rng.randn(n_trials, n_channels, n_times)
    z = rng.randn(n_trials, n_atoms, n_times_valid)
    D = rng.randn(n_atoms, n_channels, n_times_atom)

    if use_whitening:
        ar_model, X = whitening(X)
        zw = apply_whitening(ar_model, z, mode="full")
        ztz = compute_ztz(zw, n_times_atom)
        grad = np.zeros(D.shape)
        for t in range(n_times_atom):
            grad[:, :, t] = np.tensordot(ztz[:, :, t:t + n_times_atom],
                                         D[:, :, ::-1],
                                         axes=([1, 2], [0, 2]))
    else:
        ztz = compute_ztz(z, n_times_atom)
        grad = tensordot_convolve(ztz, D)
    cost = np.dot(D.ravel(), grad.ravel())

    X_hat = construct_X_multi(z, D)
    if use_whitening:
        X_hat = apply_whitening(ar_model, X_hat, mode="full")

    assert np.isclose(cost, np.dot(X_hat.ravel(), X_hat.ravel()))
