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

from alphacsc import BatchCDLfMRIFixedHRF
from alphacsc.simulate import gen_fmri_synth_data


###############################################################################
# define data
random_seed = None
tmp = gen_fmri_synth_data(_P=2000, _T=300, L=30, sigma=0.1, random_seed=None)
noisy_X, X, uv, D, u_i, v, noisy_Lz, Lz, z, cst = tmp
u_0, u_1 = u_i
T, P, p, K, N, L = cst

###############################################################################
# estimation of d an z
cdl = BatchCDLfMRIFixedHRF(K, v, reg=0.7, proba_map=True, D_init='ssa',
                           random_state=random_seed, verbose=10)
cdl.fit(noisy_X, nb_fit_try=5)
pobj, times, d_hat, z_hat = cdl.pobj_, cdl._times, cdl.uv_hat_, cdl.z_hat_

Lz_hat = np.cumsum(z_hat, axis=-1)
u_hat, v_hat = d_hat[:, :P], d_hat[0, P:]
u_0_hat = u_hat[0, :]
u_1_hat = u_hat[1, :]

###############################################################################
# plotting

# Lz
plt.figure("Temporal atoms", figsize=(5, 10))
plt.subplot(211)
plt.plot(Lz[0, :, :].T, lw=2.0)
plt.title("True blocks coding signals (Lz)")
plt.grid()
plt.subplot(212)
plt.plot(Lz_hat[0, :, :].T, '--', lw=2.0)
plt.title("Estimated blocks coding signals (Lz)")
plt.grid()

# u
plt.figure("Spatial maps", figsize=(7, 7))
plt.subplot(2, 2, 1)
plt.matshow(u_0.reshape(p, p), fignum=False)
plt.title("True spatial map 1 (u_1)")
plt.subplot(2, 2, 2)
plt.matshow(u_1.reshape(p, p), fignum=False)
plt.title("True spatial map 2 (u_2)")
plt.subplot(2, 2, 3)
plt.matshow(u_0_hat.reshape(p, p), fignum=False)
plt.title("Estimated spatial map 1 (u_1)")
plt.subplot(2, 2, 4)
plt.matshow(u_1_hat.reshape(p, p), fignum=False)
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

# times
times = times[1:]
ind = np.arange(len(times))
x_label = ['z', 'd'] * int(len(ind) / 2)

plt.figure("Benchmark duration", figsize=(5, 5))
plt.stem(times)
plt.ylabel("durations (s)")
plt.xticks(ind, x_label)
plt.title("Benchmark duration")
plt.grid()

plt.show()
