"""
=====================================================================
Simple fMRI example
=====================================================================
Example to recover the different spontanious tasks involved in the BOLD signal.
"""

# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

from alphacsc import learn_d_z_multi
from alphacsc.simulate import gen_fmri_synth_data
from alphacsc.init_dict import init_dictionary


###############################################################################
# define data
random_seed = None
tmp = gen_fmri_synth_data(random_seed=random_seed)
noisy_X, uv, D, u_i, v, noisy_Lz, Lz, cst = tmp
u_0, u_1 = u_i
T, P, K, N, L = cst

###############################################################################
# estimation of d an z
reg = 0.5
n_iter = 50
nb_runs = 5
results = []

for ii in range(1, nb_runs+1):
    print("Run: {}/{}".format(ii, nb_runs))

    uv_init = init_dictionary(noisy_X, K, L, uv_constraint='separate',
                              rank1=True, window=False, D_init='random',
                              D_init_params=dict(), random_state=random_seed)
    uv_init[:, P:] = np.repeat(v[None, :], K, axis=0)[:, :, 0]

    pobj, _, d_hat, z_hat, _ = learn_d_z_multi(
        noisy_X, K, L, reg=reg, lmbd_max='scaled', n_iter=n_iter,
        D_init=uv_init, solver_z='fista', solver_d='only_u_adaptive',
        positivity=False, random_state=random_seed,
        loss_params=dict(block=True, proba_map=True),
        raise_on_increase=False, n_jobs=1, verbose=1)

    results.append([pobj, d_hat, z_hat])

l_last_pobj = np.array([res[0][-1] for res in results])
best_run = np.argmin(l_last_pobj)
print("Best run: {}".format(best_run+1))
pobj, d_hat, z_hat = results[best_run]

Lz_hat = np.cumsum(z_hat, axis=-1)
u_hat, v_hat = d_hat[:, :P], d_hat[0, P:]
u_0_hat = u_hat[0, :]
u_1_hat = u_hat[1, :]

###############################################################################
# plotting

# Lz
plt.figure("Temporal atoms", figsize=(5, 10))
plt.subplot(211)
plt.plot(Lz[0, :, :].T / Lz[0, :, :].T.max(axis=0), lw=2.0)  # l_inf normalized
plt.title("True blocks coding signals (Lz)")
plt.grid()
plt.subplot(212)
plt.plot(Lz_hat[0, :, :].T / Lz_hat[0, :, :].T.max(axis=0), lw=2.0)
plt.title("Estimated blocks coding signals (Lz)")
plt.grid()

# u
plt.figure("Spatial maps", figsize=(7, 7))
plt.subplot(2, 2, 1)
plt.matshow(u_0.reshape(int(np.sqrt(P)), int(np.sqrt(P))), fignum=False)
plt.title("True spatial map 1 (u_1)")
plt.subplot(2, 2, 2)
plt.matshow(u_1.reshape(int(np.sqrt(P)), int(np.sqrt(P))), fignum=False)
plt.title("True spatial map 2 (u_2)")
plt.subplot(2, 2, 3)
plt.matshow(u_0_hat.reshape(int(np.sqrt(P)), int(np.sqrt(P))), fignum=False)
plt.title("Estimated spatial map 1 (u_1)")
plt.subplot(2, 2, 4)
plt.matshow(u_1_hat.reshape(int(np.sqrt(P)), int(np.sqrt(P))), fignum=False)
plt.title("Estimated spatial map 2 (u_2)")
plt.tight_layout()

# pobj
plt.figure("Cost function (%)", figsize=(5, 5))
values = np.array(pobj)
values /= (100.0 * values[0])
plt.loglog(values, lw=2.0)
plt.title("Evolution of global cost-function")
plt.xlabel('log iter')
plt.grid()

plt.show()
