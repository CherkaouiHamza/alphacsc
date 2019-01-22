"""
=====================================================================
Simple fMRI example
=====================================================================
Example to recover the different spontanious tasks involved in the BOLD signal.
"""

# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
#
# License: BSD (3-clause)

import matplotlib
matplotlib.use('Agg')

import os
import shutil
from datetime import datetime
import pickle

import numpy as np
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiMasker

from alphacsc import BatchCDLfMRIFixedHRF
from alphacsc.simulate import hrf

from utils import fetch_subject_list, get_hcp_fmri_fname, TR_HCP


###############################################################################
# results management
print(__doc__)

date = datetime.now()
dirname = ('results_hcp_'
           '#{0}{1}{2}{3}{4}{5}'.format(date.year,
                                        date.month,
                                        date.day,
                                        date.hour,
                                        date.minute,
                                        date.second))

if not os.path.exists(dirname):
    os.makedirs(dirname)

print("archiving '{0}' under '{1}'".format(__file__, dirname))
shutil.copyfile(__file__, os.path.join(dirname, __file__))

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
K = 8
L = 30
v = hrf(L)
cdl = BatchCDLfMRIFixedHRF(K, v, reg=0.7, proba_map=True, D_init='ssa',
                           random_state=random_seed, verbose=10)
cdl.fit(X, nb_fit_try=1)
pobj, times, uv_hat, z_hat = cdl.pobj_, cdl._times, cdl.uv_hat_, cdl.z_hat_

Lz_hat = np.cumsum(z_hat, axis=-1)
u_hat, v_hat = uv_hat[:, :P], uv_hat[0, P:]

###############################################################################
# archiving results
res = dict(pobj=pobj, times=times, uv_hat=uv_hat, z_hat=z_hat, Lz_hat=Lz_hat)
filename = os.path.join(dirname, "results.pkl")
print("Pickling results under '{0}'".format(filename))
with open(filename, "wb") as pfile:
    pickle.dump(res, pfile)

###############################################################################
# plotting

# Lz
plt.figure("Temporal atoms", figsize=(5, 10))
plt.plot(Lz_hat[0, :, :].T, lw=2.0)
plt.title("Estimated blocks coding signals (Lz)")
plt.grid()
filename = "Lz.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

# pobj
plt.figure("Cost function (%)", figsize=(5, 5))
values = np.array(pobj)
values /= (100.0 * values[0])
plt.loglog(values, lw=2.0)
plt.title("Evolution of global cost-function")
plt.xlabel('log iter')
plt.grid()
filename = "pobj.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)

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
filename = "times.png"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename)
