"""Convolutional dictionary learning"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>

from __future__ import print_function
import time
import sys

import numpy as np
from joblib import Parallel

from .utils import construct_X_multi, check_random_state
from .update_z_multi import update_z_multi
from .update_d_multi import update_d


def objective(X, X_hat, Z_hat, reg):
    residual = X - X_hat
    obj = 0.5 * np.sum(residual * residual) + reg * Z_hat.sum()
    return obj


def compute_X_and_objective_multi(X, Z_hat, u_hat, v_hat, reg,
                                  feasible_evaluation=True):
    d_hat = u_hat[:, :, None] * v_hat[:, None, :]
    X_hat = construct_X_multi(Z_hat, d_hat)

    if feasible_evaluation:
        Z_hat = Z_hat.copy()
        d_hat = d_hat.copy()
        # project to unit norm
        d_norm = np.linalg.norm(d_hat, axis=1)
        mask = d_norm >= 1
        d_hat[mask] /= d_norm[mask][:, None]
        # update z in the opposite way
        Z_hat[mask] *= d_norm[mask][:, None, None]

    return objective(X, X_hat, Z_hat, reg)


def learn_d_z_multi(X, n_atoms, n_times_atom, func_d=update_d, reg=0.1,
                    n_iter=60, random_state=None, n_jobs=1, solver_z='l_bfgs',
                    solver_d_kwargs=dict(), solver_z_kwargs=dict(),
                    u_init=None, v_init=None, sample_weights=None, verbose=10,
                    callback=None):
    """Learn atoms and activations using Convolutional Sparse Coding.

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times)
        The data on which to perform CSC.
    n_atoms : int
        The number of atoms to learn.
    n_times_atom : int
        The support of the atom.
    func_d : callable
        The function to update the atoms.
    reg : float
        The regularization parameter
    n_iter : int
        The number of coordinate-descent iterations.
    random_state : int | None
        The random state.
    n_jobs : int
        The number of parallel jobs.
    solver_z : str
        The solver to use for the z update. Options are
        'l_bfgs' (default) | 'ista' | 'fista'
    solver_d_kwargs : dict
        Additional keyword arguments to provide to update_d
    solver_z_kwargs : dict
        Additional keyword arguments to pass to update_z_multi
    u_init : array, shape (n_atoms, n_channels)
        The initial temporal atoms.
    v_init : array, shape (n_atoms, n_times_atoms)
        The initial temporal atoms.
    sample_weights : array, shape (n_trials, n_times)
        The weights in the alphaCSC problem. Should be None
        when using vanilla CSC.
    verbose : int
        The verbosity level.
    callback : func
        A callback function called at the end of each loop of the
        coordinate descent.

    Returns
    -------
    pobj : list
        The objective function value at each step of the coordinate descent.
    times : list
        The cumulative time for each iteration of the coordinate descent.
    u_hat : array, shape (n_atoms, n_channels)
        The estimated temporal atoms.
    v_hat : array, shape (n_atoms, n_times_atoms)
        The estimated temporal atoms.
    Z_hat : array, shape (n_atoms, n_times - n_times_atom + 1)
        The sparse activation matrix.
    """
    n_trials, n_chan, n_times = X.shape

    rng = check_random_state(random_state)

    if u_init is None:
        u_hat = rng.randn(n_atoms, n_chan, n_times_atom)
    else:
        u_hat = u_init.copy()

    if v_init is None:
        v_hat = rng.randn(n_atoms, n_chan, n_times_atom)
    else:
        v_hat = v_init.copy()

    pobj = list()
    times = list()

    if 'ista' in solver_z:
        b_hat_0 = rng.randn(n_atoms * (n_times - n_times_atom + 1))
    else:
        b_hat_0 = None

    lambd0 = None
    Z_hat = np.zeros((n_atoms, n_trials, n_times - n_times_atom + 1))

    pobj.append(
        compute_X_and_objective_multi(X, Z_hat, u_hat, v_hat, reg,
                                      sample_weights))
    times.append(0.)
    with Parallel(n_jobs=n_jobs) as parallel:
        for ii in range(n_iter):  # outer loop of coordinate descent
            if verbose == 1:
                print('.', end='')
                sys.stdout.flush()
            if verbose > 1:
                print('Coordinate descent loop %d / %d [n_jobs=%d]' %
                      (ii, n_iter, n_jobs))

            start = time.time()
            Z_hat = update_z_multi(
                X, u_hat, v_hat, reg, n_times_atom, z0=Z_hat,
                parallel=parallel, solver=solver_z, b_hat_0=b_hat_0,
                solver_kwargs=solver_z_kwargs, sample_weights=sample_weights)
            times.append(time.time() - start)

            # monitor cost function
            pobj.append(
                compute_X_and_objective_multi(X, Z_hat, u_hat, v_hat, reg,
                                              sample_weights))
            if verbose > 1:
                print('[seed %s] Objective (Z) : %0.8f' % (random_state,
                                                           pobj[-1]))

            start = time.time()
            d_hat, lambd0 = func_d(X, Z_hat, u_hat0=u_hat, v_hat0=v_hat,
                                   verbose=verbose)
            times.append(time.time() - start)

            # monitor cost function
            pobj.append(
                compute_X_and_objective_multi(X, Z_hat, d_hat, reg,
                                              sample_weights))
            if verbose > 1:
                print('[seed %s] Objective (d) %0.8f' % (random_state,
                                                         pobj[-1]))

            if callable(callback):
                callback(X, u_hat, v_hat, Z_hat, reg)

    return pobj, times, u_hat, v_hat, Z_hat
