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
from nilearn import input_data, datasets, image, plotting

from alphacsc import BatchCDLfMRIFixedHRF


###############################################################################
# results management
print(__doc__)

date = datetime.now()
dirname = 'results_ahdh_#{0}{1}{2}{3}{4}{5}'.format(date.year,
                                                    date.month,
                                                    date.day,
                                                    date.hour,
                                                    date.minute,
                                                    date.second)

if not os.path.exists(dirname):
    os.makedirs(dirname)

print("archiving '{0}' under '{1}'".format(__file__, dirname))
shutil.copyfile(__file__, os.path.join(dirname, __file__))

###############################################################################
# define data
harvard_oxford = datasets.fetch_atlas_harvard_oxford(
        'cort-maxprob-thr50-2mm', symmetric_split=True)
atlas_data = harvard_oxford.maps.get_data()
mask = atlas_data == harvard_oxford.labels.index('Right Precentral Gyrus')
mask_img = image.new_img_like(harvard_oxford.maps, mask)
adhd = datasets.fetch_adhd(n_subjects=1)
mask_img = image.resample_to_img(
    mask_img, adhd.func[0], interpolation='nearest')
masker = input_data.NiftiMasker(mask_img, t_r=2.0, standardize=True)
X = masker.fit().transform(adhd.func[0]).T
X = X[None, :, :]
N, P, T = X.shape
print("Data loaded shape: {}".format(X.shape))

###############################################################################
# estimation of d an z
random_seed = None
K = 8
L = 30
cdl = BatchCDLfMRIFixedHRF(n_atoms=K, n_times_atom=L, t_r=2.0, reg=0.2,
                           n_iter=100, proba_map=True, D_init='ssa',
                           random_state=random_seed, verbose=1)
cdl.fit(X, nb_fit_try=3)
pobj, times, uv_hat, z_hat = cdl.pobj_, cdl._times, cdl.uv_hat_, cdl.z_hat_

z_hat = z_hat[0, :, :]
u_hat = uv_hat[:, :-L]
Lz_hat = np.cumsum(z_hat, axis=-1)

###############################################################################
# archiving results
res = dict(pobj=pobj, times=times, uv_hat=uv_hat, z_hat=z_hat, Lz_hat=Lz_hat,
           L=L)
filename = os.path.join(dirname, "results.pkl")
print("Pickling results under '{0}'".format(filename))
with open(filename, "wb") as pfile:
    pickle.dump(res, pfile)

###############################################################################
# plotting
# u
for k in range(1, K+1):
    plotting.plot_glass_brain(masker.inverse_transform(u_hat[k-1]),
                              title="map-{}".format(k), colorbar=True)
    filename = "u_{0:03d}.pdf".format(k)
    filename = os.path.join(dirname, filename)
    print("Saving plot under '{0}'".format(filename))
    plt.savefig(filename, dpi=150)

# Lz
plt.figure("Temporal atoms", figsize=(10, 5*K))
for k in range(1, K+1):
    plt.subplot(K, 1, k)
    plt.plot(Lz_hat[k-1].T, lw=2.0)
    plt.title("atom-{}".format(k))
    plt.grid()
plt.tight_layout()
filename = "Lz.pdf"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename, dpi=150)

# pobj
plt.figure("Cost function (%)", figsize=(5, 5))
values = np.array(pobj)
values /= (100.0 * values[0])
plt.loglog(values, lw=2.0)
plt.title("Evolution of global cost-function")
plt.xlabel('log iter')
plt.grid()
filename = "pobj.pdf"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename, dpi=150)

# times
times = times[1:]
ind = np.arange(len(times))
x_label = ['z', 'd'] * int(len(ind) / 2)
plt.figure("Benchmark duration", figsize=(10, 5))
plt.stem(times)
plt.ylabel("durations (s)")
plt.xticks(ind, x_label)
plt.title("Benchmark duration")
plt.grid()
filename = "times.pdf"
filename = os.path.join(dirname, filename)
print("Saving plot under '{0}'".format(filename))
plt.savefig(filename, dpi=150)

import subprocess  # XXX hack to concatenate the spatial maps in one pdf
pdf_files = os.path.join(dirname, 'u_*.pdf')
pdf_file = os.path.join(dirname, 'U.pdf')
subprocess.call("pdftk {} cat output {}".format(pdf_files, pdf_file),
                shell=True)
