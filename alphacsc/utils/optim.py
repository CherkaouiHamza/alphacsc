import time
import warnings

import numpy as np
from scipy.optimize import nnls
from .compute_constants import compute_DtD
from . import check_random_state


STEP_SIZE_MIN = 1.0e-20  # XXX do not change
DEFAULT_CONVERGENCE_TOL = np.finfo(np.float32).eps
DIVERGENCE_TOL = np.finfo(np.float64).max


def fista(f_obj, f_grad, f_prox, step_size, x0, max_iter, momentum=False,
          eps=None, adaptive_step_size=False, name='ISTA',
          debug=False, timing=False, verbose=0):
    """(F)ISTA algorithm

    Parameters
    ----------
    f_obj : callable
        Objective function. Used only if debug or adaptive_step_size.
    f_grad : callable
        Gradient of the objective function
    f_prox : callable
        Proximal operator
    step_size : float or None
        Step size of each update. Can be None if adaptive_step_size.
    x0 : array
        Initial point of the optimization
    max_iter : int
        Maximum number of iterations
    verbose : int
        Verbosity level
    momentum : boolean
        If True, use FISTA instead of ISTA
    eps : float
        Tolerance for the stopping criterion
    adaptive_step_size : boolean
        If True, the step size is adapted at each step
    name : str
        Optional label for printed messages
    debug : boolean
        If True, compute the objective function at each step and return the
        list at the end.
    timing : boolean
        If True, compute the duration of each step, and return both lists at
        the end.

    Returns
    -------
    x : array
        The final point after optimization
    pobj : list or None
        If debug is True, pobj contains the value of the cost function at each
        iteration.
    times : list
        If times is True, time contians the duration for each iteration.
    """
    if timing:
        if not debug:
            warnings.warn("timing set to True: force debug to True")
        debug = True
        times = [0.0]
        start = time.time()

    pobj = None
    if debug:
        pobj = [f_obj(x0)]

    if step_size is None:
        step_size = 1.0

    _obj = None

    if eps is None:
        eps = DEFAULT_CONVERGENCE_TOL

    t = 1.0
    x_old = x0.copy()
    x = x0.copy()
    diff = x0.copy()
    grad = np.empty(x0.shape)

    for ii in range(max_iter):
        has_restarted = False
        grad[:] = f_grad(x)

        if adaptive_step_size:
            def f(step_size, x=x):
                x = f_prox(x - step_size * grad, step_size)
                return f_obj(x), x

            _obj, x, step_size = _adaptive_step_size(f, _obj, step_size)

            if step_size < STEP_SIZE_MIN:
                # We did not find a possible step size, to avoid objective
                # function to increase, we should: if FISTA then restart, if
                # ISTA then stop
                x = x_old
                if momentum:
                    has_restarted = True

        else:
            x = f_prox(x - step_size * grad, step_size)

        diff[:] = x - x_old
        x_old[:] = x

        if momentum:
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t**2))
            x[:] = x + (t - 1.0) / t_new * (diff)
            t = t_new

        l1_diff = np.sum(np.abs(diff))
        if l1_diff <= eps and not has_restarted:
            break
        if l1_diff > DIVERGENCE_TOL:
            raise RuntimeError("[{}] {} have diverged during.".format(name,
                               ["ISTA", "FISTA"][momentum]))

        if debug:
            pobj.append(f_obj(x))
            if (len(pobj) > 2) and (adaptive_step_size or (not momentum)):
                # For ISTA and FISTA with adpative step size should decrease at
                # each iteration
                d_pobj = pobj[-1] - pobj[-2]
                msg_case_1 = "ISTA"
                msg_case_2 = "FISTA with adpative step size"
                msg = (
                  "[{}] {} have increased the objective value by {:.6e}"
                  "({})".format(name, [msg_case_1, msg_case_2][momentum],
                                d_pobj, ["has restarted!", ""][has_restarted])
                    )
                if not (d_pobj < DEFAULT_CONVERGENCE_TOL):
                    warnings.warn(msg)

        if timing:
            times.append(time.time() - start)
            start = time.time()

    else:
        if verbose > 1:
            print('[{}] did not converge'.format(name))
    if verbose > 5:
        print('[{}]: {} iterations'.format(name, ii + 1))

    if timing:
        return x, pobj, times
    return x, pobj


def _adaptive_step_size(f, f0=None, alpha=None, tau=2):
    """
    Parameters
    ----------
    f : callable
        Optimized function, take only the step size as argument
    f0 : float
        value of f at current point, i.e. step size = 0
    alpha : float
        Initial step size
    tau : float
        Multiplication factor of the step size during the adaptation
    """

    if alpha is None:
        alpha = 1

    if f0 is None:
        f0, _ = f(0)
    f_alpha, x_alpha = f(alpha)
    f_alpha_down, x_alpha_down = f(alpha / tau)
    f_alpha_up, x_alpha_up = f(alpha * tau)

    alphas = [0, alpha / tau, alpha, alpha * tau]
    fs = [f0, f_alpha_down, f_alpha, f_alpha_up]
    xs = [None, x_alpha_down, x_alpha, x_alpha_up]
    i = np.argmin(fs)
    if i == 0:
        alpha /= tau * tau
        f_alpha, x_alpha = f(alpha)
        while f0 <= f_alpha and alpha > STEP_SIZE_MIN:
            alpha /= tau
            f_alpha, x_alpha = f(alpha)
        return f_alpha, x_alpha, alpha
    else:
        return fs[i], xs[i], alphas[i]


def power_iteration(lin_op, n_points=None, b_hat_0=None, max_iter=1000,
                    tol=1e-7, random_state=None):
    """Estimate dominant eigenvalue of linear operator A.

    Parameters
    ----------
    lin_op : callable or array
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
    if hasattr(lin_op, 'dot'):
        n_points = lin_op.shape[1]
        lin_op = lin_op.dot
    elif callable(lin_op):
        msg = ("power_iteration require n_points argument when lin_op is "
               "callable")
        assert n_points is not None, msg
    else:
        raise ValueError("lin_op should be a callable or a ndarray")

    rng = check_random_state(random_state)
    if b_hat_0 is None:
        b_hat = rng.rand(n_points)
    else:
        b_hat = b_hat_0

    mu_hat = np.nan
    for _ in range(max_iter):
        b_hat = lin_op(b_hat)
        norm = np.linalg.norm(b_hat)
        if norm == 0:
            return 0
        b_hat /= norm
        fb_hat = lin_op(b_hat)
        mu_old = mu_hat
        mu_hat = np.dot(b_hat, fb_hat)
        # note, we might exit the loop before b_hat converges
        # since we care only about mu_hat converging
        if (mu_hat - mu_old) / mu_old < tol:
            break

    assert not np.isnan(mu_hat)

    if b_hat_0 is not None:
        # copy inplace into b_hat_0 for next call to power_iteration
        np.copyto(b_hat_0, b_hat)

    return mu_hat


def _support_least_square(X, uv, z, debug=False):
    """WIP, not fonctional!"""
    n_trials, n_channels, n_times = X.shape
    _, _, n_times_valid = z.shape
    n_times_atom = n_times - n_times_valid + 1

    # Compute DtD
    DtD = compute_DtD(uv, n_channels)
    t0 = n_times_atom - 1
    z_hat = np.zeros(z.shape)

    for idx in range(n_trials):
        Xi = X[idx]
        support_i = z[:, idx].nonzero()
        n_support = len(support_i[0])
        if n_support == 0:
            continue
        rhs = np.zeros((n_support, n_support))
        lhs = np.zeros(n_support)

        for i, (k_i, t_i) in enumerate(zip(*support_i)):
            for j, (k_j, t_j) in enumerate(zip(*support_i)):
                dt = t_i - t_j
                if abs(dt) < n_times_atom:
                    rhs[i, j] = DtD[k_i, k_j, t0 + dt]
            aux_i = np.dot(uv[k_i, :n_channels], Xi[:, t_i:t_i + n_times_atom])
            lhs[i] = np.dot(uv[k_i, n_channels:], aux_i)

        # Solve the non-negative least-square with nnls
        z_star, _ = nnls(rhs, lhs)
        for i, (k_i, t_i) in enumerate(zip(*support_i)):
            z_hat[k_i, idx, t_i] = z_star[i]

    return z_hat
