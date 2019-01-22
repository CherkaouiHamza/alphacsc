# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Hamza Cherkaoui <hamza.cherkaoui@inria.fr>

import numpy as np

from .utils import check_random_state, construct_X
from .utils.dictionary import get_uv
from .loss_and_gradient import construct_X_multi


def hrf(n_times_atom):
        try:
            from nipype.algorithms.modelgen import spm_hrf
        except ImportError:
            raise ImportError("Please install nipype to use hrf function.")

        _hrf = spm_hrf(60./n_times_atom)
        if len(_hrf) != n_times_atom:
            # force _hrf to be n_times_atom long
            n = np.abs(len(_hrf) - n_times_atom)
            _hrf = np.hstack([_hrf, np.zeros(n)])
        return _hrf


def simulate_data(n_trials, n_times, n_times_atom, n_atoms, random_state=42,
                  constant_amplitude=False):
    """Simulate the data.

    Parameters
    ----------
    n_trials : int
        Number of samples / trials.
    n_times : int
        Number of time points.
    n_times_atom : int
        Number of time points.
    n_atoms : int
        Number of atoms.
    random_state : int | None
        If integer, fix the random state.
    constant_amplitude : float
        If True, the activations have constant amplitude.

    Returns
    -------
    X : array, shape (n_trials, n_times)
        The data
    ds : array, shape (k, n_times_atom)
        The true atoms.
    z : array, shape (n_trials, n_times - n_times_atom + 1)
        The true codes.

    Note
    ----
    X will be non-zero from n_times_atom to n_times.
    """
    # add atoms
    rng = check_random_state(random_state)
    ds = np.zeros((n_atoms, n_times_atom))
    for idx, shape, n_cycles in cycler(n_atoms, n_times_atom):
        ds[idx, :] = get_atoms(shape, n_times_atom, n_cycles=n_cycles)
    ds /= np.linalg.norm(ds, axis=1)[:, None]

    z = get_activations(rng, (n_atoms, n_trials, n_times - n_times_atom + 1),
                        constant_amplitude=constant_amplitude)
    X = construct_X(z, ds)

    assert X.shape == (n_trials, n_times)
    assert z.shape == (n_atoms, n_trials, n_times - n_times_atom + 1)
    assert ds.shape == (n_atoms, n_times_atom)

    return X, ds, z


def cycler(n_atoms, n_times_atom):
    idx = 0
    for n_cycles in range(1, n_times_atom // 2):
        for shape in ['triangle', 'square', 'sin']:
            yield idx, shape, n_cycles
            idx += 1
            if idx >= n_atoms:
                break
        if idx >= n_atoms:
            break


def get_activations(rng, shape_z, constant_amplitude=False):
    starts = list()
    n_atoms, n_trials, n_times_valid = shape_z
    for _ in range(n_atoms):
        starts.append(rng.randint(low=0, high=n_times_valid,
                      size=(n_trials,)))
    # add activations
    z = np.zeros(shape_z)
    for i in range(n_trials):
        for k_idx, _ in enumerate(starts):
            if constant_amplitude:
                randnum = 1.
            else:
                randnum = rng.uniform()
            z[k_idx, i, starts[k_idx][i]] = randnum
    return z


def get_atoms(shape, n_times_atom, zero_mean=True, n_cycles=1):
    if shape == 'triangle':
        ds = list()
        for _ in range(n_cycles):
            ds.append(np.linspace(0, 1, n_times_atom // (2 * n_cycles)))
            ds.append(ds[-1][::-1])
        d = np.hstack(ds)
        d = np.pad(d, (0, n_times_atom - d.shape[0]), 'constant')
    elif shape == 'square':
        ds = list()
        for _ in range(n_cycles):
            ds.append(0.5 * np.ones((n_times_atom // (2 * n_cycles))))
            ds.append(-ds[-1])
        d = np.hstack(ds)
        d = np.pad(d, (0, n_times_atom - d.shape[0]), 'constant')
    elif shape == 'sin':
        d = np.sin(2 * np.pi * n_cycles * np.linspace(0, 1, n_times_atom))
    elif shape == 'cos':
        d = np.cos(2 * np.pi * n_cycles * np.linspace(0, 1, n_times_atom))
    if zero_mean:
        d -= np.mean(d)

    return d


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


def gen_fmri_synth_data(_P=10000, _T=300, L=30, sigma=0.1, random_seed=None):
    """ Generate BOLD data and its corresponding spatial map.
    """
    rng = np.random.RandomState(random_seed)
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
    v = hrf(L)[:, None]
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
