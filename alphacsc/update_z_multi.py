# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Thomas Moreau <thomas.moreau@inria.fr>
#          Hamza Cherkaoui <hamza.cherkaoui@inria.fr>

import time

import numpy as np
from scipy import optimize, sparse
from joblib import Parallel, delayed


from . import cython_code
from .utils.optim import fista
from .loss_and_gradient import gradient_zi, _dense_tr_conv_d
from .utils.lil import is_list_of_lil, is_lil
from .utils.coordinate_descent import _coordinate_descent_idx
from .utils.compute_constants import compute_DtD, compute_ztz, compute_ztX


def update_z_multi(X, D, reg, z0=None, solver='l-bfgs', solver_kwargs=dict(),
                   loss='l2', loss_params=dict(), freeze_support=False,
                   positivity=True, return_ztz=False, timing=False, n_jobs=1,
                   debug=False):
    """Update z using L-BFGS/(F)ISTA/CD.

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times)
        The data array
    D : array, shape (n_atoms, n_channels + n_times_atom)
        The dictionary used to encode the signal X. Can be either in the form
        f a full rank dictionary D (n_atoms, n_channels, n_times_atom) or with
        the spatial and temporal atoms uv (n_atoms, n_channels + n_times_atom).
    reg : float
        The regularization constant
    z0 : None | array, shape (n_trials, n_atoms, n_times_valid) |
         list of sparse lil_matrices, shape (n_atoms, n_times_valid)
        Init for z (can be used for warm restart).
    solver : 'l-bfgs' | "lgcd"
        The solver to use.
    solver_kwargs : dict
        Parameters for the solver
    loss : 'l2' | 'dtw' | 'whitening'
        The data fit loss, either classical l2 norm or the soft-DTW loss.
    loss_params : dict
        Parameters of the loss
    freeze_support : boolean
        If True, the support of z0 is frozen.
    positivity : boolean
        If True set positivity constraints on z.
    return_ztz : boolean
        If True, returns the constants ztz and ztX, used to compute D-updates.
    timing : boolean
        If True, returns the cost function value at each iteration and the
        time taken by each iteration for each signal.
    n_jobs : int
        The number of parallel jobs.
    debug : bool
        If True, check the grad.

    Returns
    -------
    z : array, shape (n_trials, n_atoms, n_times - n_times_atom + 1)
        The true codes.
    """
    n_trials, n_channels, n_times = X.shape
    if D.ndim == 2:
        n_atoms, n_channels_n_times_atom = D.shape
        n_times_atom = n_channels_n_times_atom - n_channels
    else:
        n_atoms, n_channels, n_times_atom = D.shape
    n_times_valid = n_times - n_times_atom + 1

    if z0 is None:
        z0 = np.zeros((n_trials, n_atoms, n_times_valid))

    # now estimate the codes
    delayed_update_z = delayed(_update_z_multi_idx)

    results = Parallel(n_jobs=n_jobs)(
        delayed_update_z(X[i], D, reg, z0[i], debug, solver, solver_kwargs,
                         freeze_support, positivity, loss,
                         loss_params=loss_params,
                         return_ztz=return_ztz, timing=timing)
        for i in np.arange(n_trials))

    # Post process the results to get separate objects
    z_hats, pobj, times = [], [], []
    if loss == 'l2' and return_ztz:
        ztz = np.zeros((n_atoms, n_atoms, 2 * n_times_atom - 1))
        ztX = np.zeros((n_atoms, n_channels, n_times_atom))
    else:
        ztz, ztX = None, None
    for z_hat, ztz_i, ztX_i, pobj_i, times_i in results:
        z_hats.append(z_hat), pobj.append(pobj_i), times.append(times_i)
        if loss == 'l2' and return_ztz:
            ztz += ztz_i
            ztX += ztX_i

    # If z_hat is a ndarray, stack and reorder the columns
    if not is_list_of_lil(z0):
        z_hats = np.array(z_hats).reshape(n_trials, n_atoms, n_times_valid)

    if debug:
        return z_hats, ztz, ztX, pobj, times
    return z_hats, ztz, ztX


class BoundGenerator(object):
    def __init__(self, length):
        self.length = length
        self.current_index = 0

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index == self.length:
            raise StopIteration()
        self.current_index += 1
        return (0, np.inf)


def _update_z_multi_idx(X_i, D, reg, z0_i, debug, solver='l-bfgs',
                        solver_kwargs=dict(), freeze_support=False,
                        positivity=True, loss='l2', loss_params=dict(),
                        return_ztz=False, timing=False):
    t_start = time.time()
    n_channels, n_times = X_i.shape
    if D.ndim == 2:
        n_atoms, n_channels_n_times_atom = D.shape
        n_times_atom = n_channels_n_times_atom - n_channels
    else:
        n_atoms, n_channels, n_times_atom = D.shape
    n_times_valid = n_times - n_times_atom + 1

    assert not (freeze_support and z0_i is None), 'Impossible !'

    if is_lil(z0_i) and solver != "lgcd":
        raise NotImplementedError()

    constants = {}
    if solver in ("lgcd", "ista", "fista"):
        constants['DtD'] = compute_DtD(D=D, n_channels=n_channels)
        constants['DtX_i'] = _dense_tr_conv_d(X_i, D=D, n_channels=n_channels)
    init_timing = time.time() - t_start

    if z0_i is None:
        z_hat_0 = np.zeros(n_atoms * n_times_valid)
    elif is_lil(z0_i):
        z_hat_0 = z0_i
    else:
        z_hat_0 = z0_i.reshape(n_atoms * n_times_valid)

    times, pobj = None, None

    if solver == 'l-bfgs':
        def func_and_grad(zi):
            return gradient_zi(Xi=X_i, zi=zi, D=D, constants=constants,
                               reg=reg, return_func=True, flatten=True,
                               loss=loss, loss_params=loss_params)
        if timing:
            times = [init_timing]
            pobj = [func_and_grad(z_hat_0)[0]]
            t_start = [time.time()]

        if freeze_support:
            bounds = [(0, 0) if z == 0 else (0, None) for z in z_hat_0]
        else:
            bounds = BoundGenerator(n_atoms * n_times_valid)

        if not positivity:
            raise ValueError("solver_z='l-bfgs' can only be used if "
                             "'positivity=True'")
        if timing:
            def callback(xk):
                times.append(time.time() - t_start[0])
                pobj.append(func_and_grad(xk)[0])
                # use a reference to have access inside this function
                t_start[0] = time.time()
        else:
            callback = None
        factr = solver_kwargs.get('factr', 1e15)  # default value
        maxiter = solver_kwargs.get('maxiter', 15000)  # default value
        z_hat, _, _ = optimize.fmin_l_bfgs_b(
            func_and_grad, z_hat_0, fprime=None, args=(), approx_grad=False,
            bounds=bounds, factr=factr, maxiter=maxiter, callback=callback)

    elif solver in ("ista", "fista"):
        # Default args
        fista_kwargs = dict(
            max_iter=100, eps=None, verbose=0,
            momentum=(solver == "fista"),
        )
        fista_kwargs.update(solver_kwargs)

        def _obj_z(z_hat):
            return gradient_zi(Xi=X_i, zi=z_hat, D=D, constants=constants,
                               reg=reg, return_func=True, flatten=True,
                               loss=loss, loss_params=loss_params)[0]

        def _grad_z(z_hat):
            return gradient_zi(Xi=X_i, zi=z_hat, D=D, constants=constants,
                               reg=reg, return_func=False, flatten=True,
                               loss=loss, loss_params=loss_params)

        if timing:
            times = [init_timing]

        if positivity:
            def _prox_z(z_hat, step_size=1.0):
                lbda = reg * step_size
                return np.maximum(z_hat - lbda, 0.0)
        else:
            def _prox_z(z_hat, step_size=1.0):
                lbda = reg * step_size
                return np.sign(z_hat) * np.maximum(np.abs(z_hat) - lbda, 0.0)

        output = fista(_obj_z, _grad_z, _prox_z, None, z_hat_0,
                       adaptive_step_size=True, timing=timing, debug=debug,
                       name="Update z", **fista_kwargs)
        if timing:
            z_hat, pobj, times = output
            times[0] += init_timing
        else:
            z_hat, pobj = output

    elif solver == "lgcd":
        if not positivity:
                raise NotImplementedError("solver_z='lgcd' can only be used"
                                          " if 'positivity=True' (for now)")
        if not sparse.isspmatrix_lil(z_hat_0):
            z_hat_0 = z_hat_0.reshape(n_atoms, n_times_valid)

        # Default values
        tol = solver_kwargs.get('tol', 1e-1)
        n_seg = solver_kwargs.get('n_seg', 'auto')
        max_iter = solver_kwargs.get('max_iter', 1e15)
        strategy = solver_kwargs.get('strategy', 'greedy')
        output = _coordinate_descent_idx(
            X_i, D, constants, reg=reg, z0=z_hat_0,
            freeze_support=freeze_support, tol=tol, max_iter=max_iter,
            n_seg=n_seg, strategy=strategy, timing=timing, name="Update z")
        if timing:
            z_hat, pobj, times = output
            times[0] += init_timing
        else:
            z_hat = output
    else:
        raise ValueError("Unrecognized solver %s. Must be 'ista', 'fista',"
                         " 'l-bfgs', or 'lgcd'." % solver)

    if not is_lil(z_hat):
        z_hat = z_hat.reshape(n_atoms, n_times_valid)

    if loss == 'l2' and return_ztz:

        z_i = z_hat

        if not is_lil(z_hat):
            if loss_params.get('block', False):
                z_i = np.cumsum(z_hat, axis=1)
            ztz = compute_ztz(z_i[None], n_times_atom)
            ztX = compute_ztX(z_i[None], X_i[None])
        else:
            if loss_params.get('block', False):
                raise NotImplementedError("TV not implemented with lil.")
            cython_code._assert_cython()
            ztz = cython_code._fast_compute_ztz([z_i], n_times_atom)
            ztX = cython_code._fast_compute_ztX([z_i], X_i[None])
    else:
        ztz, ztX = None, None

    return z_hat, ztz, ztX, pobj, times
