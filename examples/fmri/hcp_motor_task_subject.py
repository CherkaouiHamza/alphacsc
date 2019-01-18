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
from nilearn.input_data import NiftiMasker

from alphacsc import BatchCDLfMRIFixedHRF
from alphacsc.simulate import hrf

from utils import fetch_subject_list, get_hcp_fmri_fname, TR_HCP


###############################################################################
# define data
random_seed = None
subject_id = fetch_subject_list()[0]
fmri_img = get_hcp_fmri_fname(subject_id)
X = NiftiMasker(t_r=TR_HCP, standardize=True).fit_transform(fmri_img)
N, P = X.shape
X = X.T[None, :, :]

###############################################################################
# estimation of d an z
K = 4
L = 30
v = hrf(L)
cdl = BatchCDLfMRIFixedHRF(K, v, reg=0.5, proba_map=True, D_init='ssa',
                           random_state=random_seed, verbose=10)
cdl.fit(X, nb_fit_try=1)
pobj, times, d_hat, z_hat = cdl.pobj_, cdl._times, cdl.uv_hat_, cdl.z_hat_

Lz_hat = np.cumsum(z_hat, axis=-1)
u_hat, v_hat = d_hat[:, :P], d_hat[0, P:]
u_0_hat = u_hat[0, :]
u_1_hat = u_hat[1, :]
u_2_hat = u_hat[2, :]
u_3_hat = u_hat[3, :]

###############################################################################
# plotting

# Lz
plt.figure("Temporal atoms", figsize=(5, 10))
plt.plot(Lz_hat[0, :, :].T, lw=2.0)
plt.title("Estimated blocks coding signals (Lz)")
plt.grid()

# u
plt.figure("Spatial maps", figsize=(7, 7))
plt.subplot(2, 2, 1)
plt.matshow(u_0_hat.reshape(int(np.sqrt(P)), int(np.sqrt(P))), fignum=False)
plt.title("Estimated spatial map 1 (u_0)")
plt.subplot(2, 2, 2)
plt.matshow(u_1_hat.reshape(int(np.sqrt(P)), int(np.sqrt(P))), fignum=False)
plt.title("Estimated spatial map 2 (u_1)")
plt.subplot(2, 2, 3)
plt.matshow(u_2_hat.reshape(int(np.sqrt(P)), int(np.sqrt(P))), fignum=False)
plt.title("Estimated spatial map 3 (u_2)")
plt.subplot(2, 2, 4)
plt.matshow(u_3_hat.reshape(int(np.sqrt(P)), int(np.sqrt(P))), fignum=False)
plt.title("Estimated spatial map 4 (u_3)")
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
