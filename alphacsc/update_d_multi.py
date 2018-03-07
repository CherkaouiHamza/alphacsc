"""Convolutional dictionary learning"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>

import numpy as np
from numpy import convolve
from numba import jit
import functools


from .utils import construct_X_multi, _get_D, check_random_state


PHI = (np.sqrt(5) + 1) / 2


def tensordot_convolve(ZtZ, D):
    """Compute the multivariate (valid) convolution of ZtZ and D

    Parameters
    ----------
    ZtZ: array, shape = (n_atoms, n_atoms, 2 * n_times_atom - 1)
        Activations
    D: array, shape = (n_atoms, n_channels, n_times_atom)
        Dictionnary

    Returns
    -------
    G : array, shape = (n_atoms, n_channels, n_times_atom)
        Gradient
    """
    n_atoms, n_channels, n_times_atom = D.shape
    D_revert = D[:, :, ::-1]

    G = np.zeros(D.shape)
    for t in range(n_times_atom):
        G[:, :, t] = np.tensordot(ZtZ[:, :, t:t + n_times_atom], D_revert,
                                  axes=([1, 2], [0, 2]))
    return G


def numpy_convolve_uv(ZtZ, uv):
    """Compute the multivariate (valid) convolution of ZtZ and D

    Parameters
    ----------
    ZtZ: array, shape = (n_atoms, n_atoms, 2 * n_times_atom - 1)
        Activations
    uv: array, shape = (n_atoms, n_channels + n_times_atom)
        Dictionnary

    Returns
    -------
    G : array, shape = (n_atoms, n_channels, n_times_atom)
        Gradient
    """
    assert uv.ndim == 2
    n_times_atom = (ZtZ.shape[2] + 1) // 2
    n_atoms = ZtZ.shape[0]
    n_channels = uv.shape[1] - n_times_atom

    u = uv[:, :n_channels]
    v = uv[:, n_channels:]

    G = np.zeros((n_atoms, n_channels, n_times_atom))
    for k0 in range(n_atoms):
        for k1 in range(n_atoms):
            G[k0, :, :] += (convolve(ZtZ[k0, k1], v[k1], mode='valid')[None, :]
                            * u[k1, :][:, None])

    return G


def _dense_transpose_convolve(Z, residual):
    """Convolve residual[i] with the transpose for each atom k, and return the sum

    Parameters
    ----------
    Z : array, shape (n_atoms, n_trials, n_times_valid)
    residual : array, shape (n_trials, n_chan, n_times)

    Return
    ------
    grad_D : array, shape (n_atoms, n_chan, n_times_atom)

    """
    return np.sum([[[convolve(res_ip, zik[::-1], mode='valid')  # n_times_atom
                     for res_ip in res_i]                       # n_chan
                    for zik, res_i in zip(zk, residual)]        # n_trials
                   for zk in Z], axis=1)                        # n_atoms


def _gradient_d(D, X=None, Z=None, constants=None, uv=None, n_chan=None):
    if constants:
        if D is None:
            assert uv is not None
            g = numpy_convolve_uv(constants['ZtZ'], uv)
        else:
            g = tensordot_convolve(constants['ZtZ'], D)
        return g - constants['ZtX']
    else:
        if D is None:
            assert uv is not None and n_chan is not None
            D = _get_D(uv, n_chan)
        residual = construct_X_multi(Z, D) - X
        return _dense_transpose_convolve(Z, residual)


def _gradient_uv(uv, X=None, Z=None, constants=None):
    if constants:
        n_chan = constants['n_chan']
    else:
        assert X is not None
        assert Z is not None
        n_chan = X.shape[1]
    grad_d = _gradient_d(None, X, Z, constants, uv=uv, n_chan=n_chan)
    grad_u = (grad_d * uv[:, None, n_chan:]).sum(axis=2)
    grad_v = (grad_d * uv[:, :n_chan, None]).sum(axis=1)
    return np.c_[grad_u, grad_v]


def _shifted_objective_uv(uv, constants):
    n_chan = constants['n_chan']
    grad_d = .5 * numpy_convolve_uv(constants['ZtZ'], uv) - constants['ZtX']
    cost = (grad_d * uv[:, None, n_chan:]).sum(axis=2)
    return np.dot(cost.ravel(), uv[:, :n_chan].ravel())


def prox_uv(uv, uv_constraint='joint', n_chan=None, return_norm=False):
    if uv_constraint == 'joint':
        norm_uv = np.maximum(1, np.linalg.norm(uv, axis=1))
        uv /= norm_uv[:, None]

    elif uv_constraint == 'separate':
        assert n_chan is not None
        norm_u = np.maximum(1, np.linalg.norm(uv[:, :n_chan], axis=1))
        norm_v = np.maximum(1, np.linalg.norm(uv[:, n_chan:], axis=1))

        norm_u[norm_u == 0] = 1
        norm_v[norm_v == 0] = 1
        uv[:, :n_chan] /= norm_u[:, None]
        uv[:, n_chan:] /= norm_v[:, None]
        norm_uv = norm_u * norm_v
    else:
        raise ValueError('Unknown uv_constraint: %s.' % (uv_constraint, ))

    if return_norm:
        return uv, norm_uv
    else:
        return uv


def update_uv(X, Z, uv_hat0, b_hat_0=None, debug=False, max_iter=300, eps=None,
              momentum=False, uv_constraint='joint', verbose=0):
    """Learn d's in time domain.

    Parameters
    ----------
    X : array, shape (n_trials, n_times)
        The data for sparse coding
    Z : array, shape (n_atoms, n_trials, n_times - n_times_atom + 1)
        The code for which to learn the atoms
    uv_hat0 : array, shape (n_atoms, n_channels + n_times_atom)
        The initial atoms.
    b_hat_0 : array, shape (n_atoms * (n_channels + n_times_atom))
        Init eigen-vector vector used in power_iteration, used in warm start.
    debug : bool
        If True, return the cost at each iteration.
    momentum : bool
        If True, use an accelerated version of the proximal gradient descent.
    uv_constraint : str in {'joint', 'separate', 'box'}
        The kind of norm constraint on the atoms:
        If 'joint', the constraint is norm_2([u, v]) <= 1
        If 'separate', the constraint is norm_2(u) <= 1 and norm_2(v) <= 1
        If 'box', the constraint is norm_inf([u, v]) <= 1
    verbose : int
        Verbosity level.

    Returns
    -------
    uv_hat : array, shape (n_atoms, n_channels + n_times_atom)
        The atoms to learn from the data.
    """
    n_atoms, n_trials, n_times_valid = Z.shape
    _, n_chan, n_times = X.shape

    # XXX : FISTA does not work and the cost goes up, should be fixed.
    constants = _get_d_update_constants(X, Z, b_hat_0=b_hat_0)

    def objective(uv, full=False):
        cost = _shifted_objective_uv(uv, constants)
        if full:
            cost += np.sum(X * X) / 2
        return cost

    def gradient(uv):
        uv = uv.reshape((n_atoms, -1))
        return _gradient_uv(uv, constants=constants).ravel()

    if uv_constraint == 'joint':
        # TODO: add a line-search

        if debug:
            pobj = list()

        if eps is None:
            eps = np.finfo(np.float32).eps
        tk = 1
        uv_hat = uv_hat0.copy()
        uv_hat_aux = uv_hat.copy()
        grad = np.empty(uv_hat.shape)
        diff = np.empty(uv_hat.shape)
        alpha = None
        for ii in range(max_iter):

            grad[:] = _gradient_uv(uv_hat_aux, constants=constants)
            alpha = _line_search(objective, uv_hat_aux, grad, alpha=alpha)
            uv_hat_aux -= alpha * grad
            prox_uv(uv_hat_aux, uv_constraint=uv_constraint, n_chan=n_chan)
            diff[:] = uv_hat_aux - uv_hat
            uv_hat[:] = uv_hat_aux
            if momentum:  # TODO: FISTA does not work well!
                tk_new = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
                uv_hat_aux += (tk - 1) / tk_new * diff
                tk = tk_new
            if debug:
                pobj.append(objective(uv_hat))
            f = np.sum(abs(diff))
            if f <= eps:
                break
            if f > 1e50:
                raise RuntimeError("The D update have diverged.")
        else:
            if verbose > 1:
                print('update_uv did not converge')
        if verbose > 1:
            print('%d iterations' % (ii + 1))

    elif uv_constraint == 'separate':
        # TODO add a for loop on u then a for loop on v
        # (need to compute Lu and Lv with two power_iteration runs)
        raise NotImplementedError
    elif uv_constraint == 'box':
        # TODO use l_bfgs_b solver on a L_inf joint box constraint
        raise NotImplementedError
    else:
        raise ValueError('Unknown uv_constraint: %s' % (uv_constraint, ))

    if debug:
        return uv_hat, pobj
    return uv_hat


def _get_d_update_constants(X, Z, b_hat_0=None):
    # Get shapes
    n_atoms, n_trials, n_times_valid = Z.shape
    _, n_chan, n_times = X.shape
    n_times_atom = n_times - n_times_valid + 1

    constants = {}
    constants['ZtX'] = np.sum(
        [[[convolve(zik[::-1], xip, mode='valid') for xip in xi]
          for zik, xi in zip(zk, X)] for zk in Z], axis=1)

    ZtZ = compute_ZtZ(Z, n_times_atom)
    constants['ZtZ'] = ZtZ
    constants['n_chan'] = n_chan

    def op_H(uv):
        uv = uv.reshape(n_atoms, n_chan + n_times_atom)
        H_d = 3 * numpy_convolve_uv(ZtZ, uv) - constants['ZtX']
        H_u = (H_d * uv[:, None, n_chan:]).sum(axis=2)
        H_v = (H_d * uv[:, :n_chan, None]).sum(axis=1)
        return np.c_[H_u, H_v].flatten()

    n_points = n_atoms * (n_chan + n_times_atom)
    constants['L'] = power_iteration(op_H, n_points, b_hat_0=b_hat_0)

    return constants


@jit()
def compute_ZtZ(Z, n_times_atom):
    """
    ZtZ.shape = n_atoms, n_atoms, 2 * n_times_atom - 1
    Z.shape = n_atoms, n_trials, n_times - n_times_atom + 1)
    """
    n_atoms, n_trials, n_times_valid = Z.shape

    ZtZ = np.zeros(shape=(n_atoms, n_atoms, 2 * n_times_atom - 1))
    t0 = n_times_atom - 1
    for k0 in range(n_atoms):
        for k in range(n_atoms):
            for i in range(n_trials):
                for t in range(n_times_atom):
                    if t == 0:
                        ZtZ[k0, k, t0] += (Z[k0, i] * Z[k, i]).sum()
                    else:
                        ZtZ[k0, k, t0 + t] += (
                            Z[k0, i, :-t] * Z[k, i, t:]).sum()
                        ZtZ[k0, k, t0 - t] += (
                            Z[k0, i, t:] * Z[k, i, :-t]).sum()
    return ZtZ


def power_iteration(lin_op, n_points, b_hat_0=None, max_iter=1000, tol=1e-7,
                    random_state=None):
    """Estimate dominant eigenvalue of linear operator A.

    Parameters
    ----------
    lin_op : callable
        Linear operator from which we estimate the largest eigenvalue.
    n_points : tuple
        Input shape of the linear operator `lin_op`.
    b_hat_0 : array, shape (n_points, )
        Init vector. The estimated eigen-vector is stored inplace in `b_hat_0`
        to allow warm start of future call of this function with the same
        variable.

    Returns
    -------
    mu_hat : float
        The largest eigenvalue
    """
    rng = check_random_state(random_state)
    if b_hat_0 is None:
        b_hat = rng.rand(n_points)
    else:
        b_hat = b_hat_0

    mu_hat = np.nan
    for ii in range(max_iter):
        b_hat = lin_op(b_hat)
        b_hat /= np.linalg.norm(b_hat)
        fb_hat = lin_op(b_hat)
        mu_old = mu_hat
        mu_hat = np.dot(b_hat, fb_hat)
        # note, we might exit the loop before b_hat converges
        # since we care only about mu_hat converging
        if (mu_hat - mu_old) / mu_old < tol:
            break

    if b_hat_0 is not None:
        # copy inplace into b_hat_0 for next call to power_iteration
        np.copyto(b_hat_0, b_hat)

    return mu_hat


def _line_search(objective, xk, gk, f0=None, alpha=None, tau=1.2, tol=1e-5):

    if f0 is None:
        f0 = objective(xk)
    if alpha is None or True:
        alpha = 1e10

    @functools.lru_cache(maxsize=None)
    def f(step_size):
        return objective(xk - step_size * gk)

    alpha1 = alpha
    f_alpha = f(alpha)

    # Find the smallest alpha with f(alpha) >= f0
    if f_alpha < f0:
        alpha1 *= tau
        while f(alpha1) < f0:
            alpha1 *= tau
    while f(alpha1 / tau) > f0:
        alpha1 /= tau

    alpha0 = 1e-25

    c = alpha1 - (alpha1 - alpha0) / PHI
    d = alpha0 + (alpha1 - alpha0) / PHI
    i = 0
    while abs(c - d) > tol and abs(f(c) - f(d)) > tol:
        if f(c) < f(d):
            alpha1 = d
        else:
            alpha0 = c

        # we recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = alpha1 - (alpha1 - alpha0) / PHI
        d = alpha0 + (alpha1 - alpha0) / PHI
        i += 1


    try:
        assert f0 >= f(c) or f0 >= f(alpha0)
    except AssertionError:
        import IPython, sys
        IPython.embed()
        sys.exit()

    if f((alpha0 + alpha1) / 2) < f0:
        return (alpha0 + alpha1) / 2
    if f(c) < f0:
        return c
    assert f(alpha0) <= f0
    return alpha0
