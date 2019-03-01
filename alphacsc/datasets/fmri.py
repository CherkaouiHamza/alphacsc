# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>

import numpy as np

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


def _gen_single_case_checkerboard(p, n=5):
    """ Return a matrice of 0.0 with only one block one 1.0. """
    d = int(p/n)
    b = (slice(d, 2*d), slice(d*(n-2), d*(n-1)))
    u = np.zeros((p, p))
    u[b] = 1.0
    return u


def _blocks_signal(T=300, n=10, rng=np.random):
    """ return a 1d signal of n blocks of length T
    """
    d = int(T / n)
    s = []
    for _ in range(n // 2):
        a = 1.0 + np.abs(0.5 * rng.randn())
        s += [a] * d + [0.0] * d
    s += [0.0] * (n % 2)
    return np.array(s)


def _2_blocks_signal(T=300, n=10, rng=np.random):
    """ return a 1d signal of n blocks of length T
    """
    d = int(T/n)
    b_11 = slice(d, 2*d)
    b_12 = slice(d*(n-4), d*(n-3))
    b_21 = slice(3*d, 4*d)
    b_22 = slice(d*(n-2), d*(n-1))
    Lz_0 = np.zeros(T)
    Lz_1 = np.zeros(T)
    Lz_0[b_11] = 1.0 + np.abs(0.5 * rng.randn())
    Lz_0[b_12] = 1.0 + np.abs(0.5 * rng.randn())
    Lz_1[b_21] = 1.0 + np.abs(0.5 * rng.randn())
    Lz_1[b_22] = 1.0 + np.abs(0.5 * rng.randn())
    return Lz_0, Lz_1


def add_gaussian_noise(signal, snr, random_state=None):
    """ Add a Gaussian noise as targeted by the given SNR value.
    """
    rng = check_random_state(random_state)
    noise = rng.randn(*signal.shape)
    true_snr_num = np.linalg.norm(signal)
    true_snr_deno = np.linalg.norm(noise)
    true_snr = true_snr_num / (true_snr_deno + np.finfo(np.float).eps)
    std_dev = (1.0 / np.sqrt(10**(snr/10.0))) * true_snr
    noise = std_dev * noise
    noisy_signal = signal + noise
    return noisy_signal, noise, std_dev


def load_synth_fmri_data(_P=100, _T=100, L=20, nb_square=2, snr=1.0,
                         eta=10.0, random_seed=None):
    """ Generate BOLD data and its corresponding spatial map.
    """
    # eta fix to 10.0 to match the actuel model in alphacsc

    rng = check_random_state(random_seed)
    N, K = 1, 2

    # z
    Lz_0, Lz_1 = _2_blocks_signal(T=_T, rng=rng)
    Lz_0 -= np.mean(Lz_0)
    Lz_1 -= np.mean(Lz_1)
    z_0 = np.append(np.diff(Lz_0), 0)
    z_1 = np.append(np.diff(Lz_1), 0)
    Lz = np.vstack([Lz_0, Lz_1])[None]
    z = np.vstack([z_0, z_1])[None]
    T = Lz_0.shape[0]

    # v
    v = spm_hrf(1.0, L)[:, None]
    v /= np.linalg.norm(v)

    # normalized each u_i
    p = int(np.sqrt(_P))
    u_0 = _gen_single_case_checkerboard(p)
    u_1 = u_0.T
    u_0 = u_0.flatten()[:, None]
    u_1 = u_1.flatten()[:, None]
    u_0 *= (eta / np.sum(np.abs(u_0)))
    u_1 *= (eta / np.sum(np.abs(u_1)))
    u_i = [u_0, u_1]
    P = u_0.size

    # D uv
    uv_0 = u_0.dot(v.T)
    uv_1 = u_1.dot(v.T)
    D = np.stack([uv_0, uv_1])
    uv = get_uv(D)

    # check dimension
    assert Lz.shape == (N, K, T)
    assert z.shape == (N, K, T)
    assert v.shape == (L, 1)
    assert uv.shape == (K, P+L)
    assert D.shape == (K, P, L)
    assert P == p*p
    cst = [T, P, p, K, N, L]

    # X
    X = construct_X_multi(Lz, D, n_channels=P)
    noisy_X, _, _ = add_gaussian_noise(X, snr, random_state=random_seed)

    return noisy_X, X, uv, D, u_i, v, Lz, z, cst
