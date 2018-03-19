"""Convolutional dictionary learning"""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>

from __future__ import print_function
import time
import sys

import numpy as np
from joblib import Parallel

from .utils import construct_X_multi_uv, check_random_state
from .utils import _patch_reconstruction_error, _get_uv
from .update_z_multi import update_z_multi
from .update_d_multi import update_uv, prox_uv
from .init_dict import init_uv
from .profile_this import profile_this  # noqa


def objective(X, X_hat, Z_hat, reg):
    residual = X - X_hat
    obj = 0.5 * np.sum(residual * residual) + reg * Z_hat.sum()
    return obj


def compute_X_and_objective_multi(X, Z_hat, uv_hat, reg,
                                  feasible_evaluation=True,
                                  uv_constraint='joint'):
    """Compute X and return the value of the objective function

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times)
        The data on which to perform CSC.
    Z_hat : array, shape (n_atoms, n_times - n_times_atom + 1)
        The sparse activation matrix.
    uv_hat : array, shape (n_atoms, n_channels + n_times_atom)
        The atoms to learn from the data.
    reg : float
        The regularization Parameters
    feasible_evaluation: boolean
        If feasible_evaluation is True, it first projects on the feasible set,
        i.e. norm(uv_hat) <= 1.
    uv_constraint : str in {'joint', 'separate'}
        The kind of norm constraint on the atoms:
        If 'joint', the constraint is norm([u, v]) <= 1
        If 'separate', the constraint is norm(u) <= 1 and norm(v) <= 1
    """
    n_chan = X.shape[1]

    if feasible_evaluation:
        Z_hat = Z_hat.copy()
        uv_hat = uv_hat.copy()
        # project to unit norm
        uv_hat, norm_uv = prox_uv(uv_hat, uv_constraint=uv_constraint,
                                  n_chan=n_chan, return_norm=True)
        # update z in the opposite way
        Z_hat *= norm_uv[:, None, None]

    X_hat = construct_X_multi_uv(Z_hat, uv_hat, n_chan)

    return objective(X, X_hat, Z_hat, reg)


# @profile_this
def learn_d_z_multi(X, n_atoms, n_times_atom, func_d=update_uv, reg=0.1,
                    n_iter=60, random_state=None, n_jobs=1, solver_z='l_bfgs',
                    solver_d='alternate', uv_constraint='separate',
                    solver_d_kwargs=dict(), solver_z_kwargs=dict(),
                    eps=1e-10, uv_init=None, verbose=10, callback=None,
                    stopping_pobj=None, kmeans_params=dict()):
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
    solver_d : str
        The solver to use for the d update. Options are
        'alternate' (default) | 'joint' | 'lbfgs'
    uv_constraint : str in {'joint', 'separate', 'box'}
        The kind of norm constraint on the atoms:
        If 'joint', the constraint is norm_2([u, v]) <= 1
        If 'separate', the constraint is norm_2(u) <= 1 and norm_2(v) <= 1
        If 'box', the constraint is norm_inf([u, v]) <= 1
    solver_d_kwargs : dict
        Additional keyword arguments to provide to update_d
    solver_z_kwargs : dict
        Additional keyword arguments to pass to update_z_multi
    uv_init : array, shape (n_atoms, n_channels + n_times_atoms)
        The initial atoms.
    verbose : int
        The verbosity level.
    callback : func
        A callback function called at the end of each loop of the
        coordinate descent.
    kmeans_params : dict
        Dictionnary of parameters for the kmeans init method.

    Returns
    -------
    pobj : list
        The objective function value at each step of the coordinate descent.
    times : list
        The cumulative time for each iteration of the coordinate descent.
    uv_hat : array, shape (n_atoms, n_channels + n_times_atom)
        The atoms to learn from the data.
    Z_hat : array, shape (n_atoms, n_trials, n_times_valid)
        The sparse activation matrix.
    """
    n_trials, n_chan, n_times = X.shape
    n_times_valid = n_times - n_times_atom + 1

    pobj = list()
    times = list()

    # initialization
    start = time.time()
    rng = check_random_state(random_state)

    uv_hat = init_uv(X, n_atoms, n_times_atom, uv_init=uv_init,
                     uv_constraint=uv_constraint, random_state=rng,
                     kmeans_params=kmeans_params)
    b_hat_0 = rng.randn(n_atoms * (n_chan + n_times_atom))
    times.append(time.time() - start)

    Z_hat = np.zeros((n_atoms, n_trials, n_times_valid))
    pobj.append(compute_X_and_objective_multi(X, Z_hat, uv_hat, reg,
                uv_constraint=uv_constraint))

    with Parallel(n_jobs=n_jobs) as parallel:
        for ii in range(n_iter):  # outer loop of coordinate descent
            if verbose == 1:
                print('.', end='')
                sys.stdout.flush()
            if verbose > 1:
                print('Coordinate descent loop %d / %d [n_jobs=%d]' %
                      (ii, n_iter, n_jobs))

            start = time.time()
            z_kwargs = dict(verbose=verbose, eps=1e-8)
            z_kwargs.update(solver_z_kwargs)
            Z_hat = update_z_multi(
                X, uv_hat, reg=reg, z0=Z_hat, parallel=parallel,
                solver=solver_z, solver_kwargs=z_kwargs)
            times.append(time.time() - start)

            if len(Z_hat.nonzero()[0]) == 0:
                import warnings
                warnings.warn("Regularization parameter `reg` is too large "
                              "and all the activations are zero. No atoms has"
                              " been learned.", UserWarning)
                break

            if verbose > 1:
                print("sparsity:", np.sum(Z_hat != 0) / Z_hat.size)

            # monitor cost function
            pobj.append(compute_X_and_objective_multi(X, Z_hat, uv_hat, reg,
                        uv_constraint=uv_constraint))
            if verbose > 1:
                print('[seed %s] Objective (Z) : %0.4e' % (random_state,
                                                           pobj[-1]))

            start = time.time()
            d_kwargs = dict(verbose=verbose, eps=1e-8)
            d_kwargs.update(solver_d_kwargs)

            uv_hat = func_d(X, Z_hat, uv_hat0=uv_hat, b_hat_0=b_hat_0,
                            solver_d=solver_d, uv_constraint=uv_constraint,
                            **d_kwargs)
            times.append(time.time() - start)

            pobj.append(compute_X_and_objective_multi(X, Z_hat, uv_hat, reg,
                        uv_constraint=uv_constraint))

            null_atom_indices = np.where(abs(Z_hat).sum(axis=(1, 2)) == 0)[0]
            if len(null_atom_indices) > 0:
                k0 = null_atom_indices[0]
                uv_hat[k0] = get_max_error_dict(X, Z_hat, uv_hat)[0]
                if verbose > 1:
                    print('[seed %s] Resampled atom %d' % (random_state, k0))

            if verbose > 1:
                print('[seed %s] Objective (d) : %0.4e' % (random_state,
                                                           pobj[-1]))

            if callable(callback):
                callback(X, uv_hat, Z_hat, reg)

            if pobj[-3] - pobj[-2] < eps and pobj[-2] - pobj[-1] < eps:
                if pobj[-3] - pobj[-2] < -eps:
                    raise RuntimeError(
                        "The z update have increased the objective value.")
                if pobj[-2] - pobj[-1] < -eps:
                    raise RuntimeError(
                        "The d update have increased the objective value.")
                break

            if stopping_pobj is not None and pobj[-1] < stopping_pobj:
                break

    # recompute Z_hat with no regularization and keeping the support fixed
    Z_hat = update_z_multi(
        X, uv_hat, reg=0, z0=Z_hat, parallel=parallel,
        solver=solver_z, solver_kwargs=solver_z_kwargs, freeze_support=True)

    return pobj, times, uv_hat, Z_hat


def get_max_error_dict(X, Z, uv):
    """Get the maximal reconstruction error patch from the data as a new atom

    This idea is used for instance in [Yellin2017]

    Parameters
    ----------
    X: array, shape (n_trials, n_channels, n_times)
        Signals encoded in the CSC.
    Z: array, shape (n_atoms, n_trials, n_times_valid)
        Current estimate of the coding signals.
    uv: array, shape (n_atoms, n_channels + n_times_atom)
        Current estimate of the rank1 multivariate dictionary.

    Return
    ------
    uvk: array, shape (n_channels + n_times_atom,)
        New atom for the dictionary, chosen as the chunk of data with the
        maximal reconstruction error.

    [Yellin2017] BLOOD CELL DETECTION AND COUNTING IN HOLOGRAPHIC LENS-FREE 
    IMAGING BY CONVOLUTIONAL SPARSE DICTIONARY LEARNING AND CODING.
    """
    n_trials, n_channels, n_times = X.shape
    n_times_atom = uv.shape[1] - n_channels
    patch_rec_error = _patch_reconstruction_error(X, Z, uv)
    i0 = patch_rec_error.argmax()
    n0, t0 = np.unravel_index(i0, Z.shape[1:])

    d0 = X[n0, :, t0:t0 + n_times_atom][None]

    return _get_uv(d0)

@profile_this
def learn_uv_z(X, n_atoms, n_times_atom, func_d=update_uv, reg=0.1,
               n_iter=60, random_state=None, n_jobs=1, solver_z='l_bfgs',
               solver_d='alternate', uv_constraint='separate',
               solver_d_kwargs=dict(), solver_z_kwargs=dict(),
               eps=1e-10, uv_init=None, verbose=10, callback=None,
               stopping_pobj=None, kmeans_params=dict(), algorithm='batch'):
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
    solver_d : str
        The solver to use for the d update. Options are
        'alternate' (default) | 'joint' | 'lbfgs'
    uv_constraint : str in {'joint', 'separate', 'box'}
        The kind of norm constraint on the atoms:
        If 'joint', the constraint is norm_2([u, v]) <= 1
        If 'separate', the constraint is norm_2(u) <= 1 and norm_2(v) <= 1
        If 'box', the constraint is norm_inf([u, v]) <= 1
    solver_d_kwargs : dict
        Additional keyword arguments to provide to update_d
    solver_z_kwargs : dict
        Additional keyword arguments to pass to update_z_multi
    uv_init : array, shape (n_atoms, n_channels + n_times_atoms)
        The initial atoms.
    verbose : int
        The verbosity level.
    callback : func
        A callback function called at the end of each loop of the
        coordinate descent.
    kmeans_params : dict
        Dictionnary of parameters for the kmeans init method.

    Returns
    -------
    pobj : list
        The objective function value at each step of the coordinate descent.
    times : list
        The cumulative time for each iteration of the coordinate descent.
    uv_hat : array, shape (n_atoms, n_channels + n_times_atom)
        The atoms to learn from the data.
    Z_hat : array, shape (n_atoms, n_trials, n_times_valid)
        The sparse activation matrix.
    """
    n_trials, n_chan, n_times = X.shape
    n_times_valid = n_times - n_times_atom + 1

    pobj = list()
    times = list()

    # initialization
    start = time.time()
    rng = check_random_state(random_state)

    uv_hat = init_uv(X, n_atoms, n_times_atom, uv_init=uv_init,
                     uv_constraint=uv_constraint, random_state=rng,
                     kmeans_params=kmeans_params)
    b_hat_0 = rng.randn(n_atoms * (n_chan + n_times_atom))
    times.append(time.time() - start)

    Z_hat = np.zeros((n_atoms, n_trials, n_times_valid))

    z_kwargs = dict(verbose=verbose)
    z_kwargs.update(solver_z_kwargs)

    def compute_z_func(X, Z_hat, uv_hat, parallel=None):
        return update_z_multi(X, uv_hat, reg=reg, z0=Z_hat, parallel=parallel,
                              solver=solver_z, solver_kwargs=z_kwargs)

    def obj_func(X, Z_hat, uv_hat):
        return compute_X_and_objective_multi(X, Z_hat, uv_hat, reg,
                                             uv_constraint=uv_constraint)

    d_kwargs = dict(verbose=verbose, eps=1e-8)
    d_kwargs.update(solver_d_kwargs)

    def compute_d_func(X, Z_hat, uv_hat):
        return func_d(X, Z_hat, uv_hat0=uv_hat, b_hat_0=b_hat_0,
                      solver_d=solver_d, uv_constraint=uv_constraint,
                      **d_kwargs)

    end_iter_func = get_iteration_func(reg, eps, stopping_pobj, callback)

    with Parallel(n_jobs=n_jobs) as parallel:
        if algorithm == 'batch':
            pobj, times, uv_hat, Z_hat = _batch_learn(
                X, uv_hat, Z_hat, compute_z_func, compute_d_func, obj_func,
                end_iter_func, n_iter=n_iter, n_jobs=n_jobs, verbose=verbose,
                random_state=random_state, parallel=parallel
            )
        elif algorithm == "greedy":
            pobj, times, uv_hat, Z_hat = _batch_learn(
                X, uv_hat, Z_hat, compute_z_func, compute_d_func, obj_func,
                end_iter_func, n_iter=n_iter, n_jobs=n_jobs, verbose=verbose,
                random_state=random_state, parallel=parallel
            )
        else:
            raise NotImplementedError("Algorithm {} is not implemented to "
                                      "dictionary atoms.".format(algorithm))

        # recompute Z_hat with no regularization and keeping the support fixed
        Z_hat = update_z_multi(
            X, uv_hat, reg=0, z0=Z_hat, parallel=parallel, solver=solver_z,
            solver_kwargs=solver_z_kwargs, freeze_support=True)

    return pobj, times, uv_hat, Z_hat


def _batch_learn(X, uv_hat, Z_hat, compute_z_func, compute_d_func,
                 obj_func, end_iter_func, n_iter=100, n_jobs=1, verbose=0,
                 random_state=None, parallel=None):

    pobj = list()
    times = list()

    # monitor cost function
    pobj.append(obj_func(X, Z_hat, uv_hat))

    for ii in range(n_iter):  # outer loop of coordinate descent
        if verbose == 1:
            print('.', end='')
            sys.stdout.flush()
        if verbose > 1:
            print('Coordinate descent loop %d / %d [n_jobs=%d]' %
                  (ii, n_iter, n_jobs))

        start = time.time()
        Z_hat = compute_z_func(X, Z_hat, uv_hat, parallel=parallel)
        times.append(time.time() - start)

        if len(Z_hat.nonzero()[0]) == 0:
            import warnings
            warnings.warn("Regularization parameter `reg` is too large "
                          "and all the activations are zero. No atoms has"
                          " been learned.", UserWarning)
            break

        if verbose > 1:
            print("sparsity:", np.sum(Z_hat != 0) / Z_hat.size)

        # monitor cost function
        pobj.append(obj_func(X, Z_hat, uv_hat))
        if verbose > 1:
            print('[seed %s] Objective (Z) : %0.4e' % (random_state,
                                                       pobj[-1]))

        start = time.time()

        uv_hat = compute_d_func(X, Z_hat, uv_hat)
        times.append(time.time() - start)

        pobj.append(obj_func(X, Z_hat, uv_hat))

        null_atom_indices = np.where(abs(Z_hat).sum(axis=(1, 2)) == 0)[0]
        if len(null_atom_indices) > 0:
            k0 = null_atom_indices[0]
            uv_hat[k0] = get_max_error_dict(X, Z_hat, uv_hat)[0]
            if verbose > 1:
                print('[seed %s] Resampled atom %d' % (random_state, k0))

        if verbose > 1:
            print('[seed %s] Objective (d) : %0.4e' % (random_state,
                                                       pobj[-1]))

        if end_iter_func(X, Z_hat, uv_hat, pobj):
            break

    return pobj, times, uv_hat, Z_hat


def get_iteration_func(reg, eps, stopping_pobj, callback):
    def end_iteration(X, Z_hat, uv_hat, pobj):
        if callable(callback):
            callback(X, uv_hat, Z_hat, reg)

        if (pobj[-3] - pobj[-2] < eps * pobj[-1] and
                pobj[-2] - pobj[-1] < eps * pobj[-1]):
            if pobj[-3] - pobj[-2] < -eps:
                raise RuntimeError(
                    "The z update have increased the objective value.")
            if pobj[-2] - pobj[-1] < -eps:
                raise RuntimeError(
                    "The d update have increased the objective value.")
            return True

        if stopping_pobj is not None and pobj[-1] < stopping_pobj:
            return True
        return False

    return end_iteration
