# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>

import os
import tempfile

import numpy as np
from scipy import io, sparse

from ..utils import check_random_state
from ..utils.dictionary import get_uv
from ..utils.hrf_model import spm_hrf
from ..loss_and_gradient import construct_X_multi


def _gen_checkerboard(n_r=5, n_c=5, s=10):
    """ return a [0, 1] cherckerboard array of 2*n_r squares of dim s in each
    rows and 2*n_c squares of dim s in each cols
    """
    assert n_r != 0 and n_c != 0
    return np.kron([[1.0, 0.0] * n_c, [0.0, 1.0] * n_c] * n_r, np.ones((s, s)))


def _blocks_signal(T=300, n=10):
    """ return a 1d signal of n blocks of length T
    """
    d = int(T / n)
    return np.array(([1.0] * d + [0.0] * d) * (n // 2) + [0.0] * (n % 2))


def load_synth_fmri_data(_P=10000, _T=300, L=30, sigma=0.1, random_seed=None):
    """ Generate BOLD data and its corresponding spatial map.
    """
    rng = check_random_state(random_seed)
    N, K = 1, 2

    # z
    Lz_0 = _blocks_signal(_T, n=10)
    Lz_1 = np.abs(Lz_0 - 1.0)
    z_0 = np.append(np.diff(Lz_0), 0)
    z_1 = np.append(np.diff(Lz_1), 0)
    Lz = np.vstack([Lz_0, Lz_1])[None]
    z = np.vstack([z_0, z_1])[None]
    T = Lz_0.shape[0]

    # add noise on z
    noisy_Lz = Lz + sigma * rng.randn(*Lz.shape)

    # v
    v = spm_hrf(1.0, L)[:, None]
    v /= np.linalg.norm(v)

    # normalized each u_i
    s = int(np.sqrt(_P) / 10.0)
    assert s != 0, "gen_fmri_synth_data called with P too small"
    n_r = n_c = int(np.sqrt(_P) / (2 * s))
    p = 2 * n_r * s

    u_0 = _gen_checkerboard(n_r, n_c, s)
    u_0 = u_0.flatten()[:, None]
    u_1 = np.abs(u_0 - 1.0)
    u_0 /= np.linalg.norm(u_0)
    u_1 /= np.linalg.norm(u_1)
    u_i = [u_0, u_1]
    P = u_0.size

    # D uv
    uv_0 = u_0.dot(v.T)
    uv_1 = u_1.dot(v.T)
    D = np.stack([uv_0, uv_1])
    uv = get_uv(D)

    assert Lz.shape == (N, K, T)
    assert z.shape == (N, K, T)
    assert v.shape == (L, 1)
    assert uv.shape == (K, P+L)
    assert D.shape == (K, P, L)
    assert P == p*p
    cst = [T, P, p, K, N, L]

    # X
    noisy_X = construct_X_multi(noisy_Lz, D, n_channels=P)
    X = construct_X_multi(Lz, D, n_channels=P)

    return noisy_X, X, uv, D, u_i, v, noisy_Lz, Lz, z, cst
