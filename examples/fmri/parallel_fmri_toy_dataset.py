""" Simple fMRI example: example to recover the different spontanious tasks
involved in the BOLD signal."""
# Authors: Hamza Cherkaoui <hamza.cherkaoui@inria.fr>
# License: BSD (3-clause)

# import matplotlib
# matplotlib.use('Agg')

import os
import shutil
import subprocess
import itertools
from datetime import datetime
import pickle

import numpy as np
import matplotlib.pyplot as plt

from alphacsc import BatchCDLfMRIFixedHRF
from alphacsc.datasets.fmri import load_synth_fmri_data


###############################################################################
# results management
print(__doc__)

date = datetime.now()
dirname = 'results_toy_#{0}{1}{2}{3}{4}{5}'.format(date.year, date.month,
                                                   date.day, date.hour,
                                                   date.minute, date.second)

if not os.path.exists(dirname):
    os.makedirs(dirname)

print("archiving '{0}' under '{1}'".format(__file__, dirname))
shutil.copyfile(__file__, os.path.join(dirname, __file__))

###############################################################################
# define data
tmp = load_synth_fmri_data(_P=100, _T=100, L=30, snr=1.0, nb_square=4,
                           random_seed=None)
noisy_X, X, uv, D, u_i, v, Lz, z, cst = tmp
u_0, u_1 = u_i
T, P, p, K, N, L = cst

###############################################################################
# estimation of u an z
l_reg_z = np.linspace(0.0, 0.7, 5)
l_reg_u = np.linspace(0.0, 10.0, 5)
l_reg = list(itertools.product(l_reg_z, l_reg_u))
l_err = []
l_res = []
for reg_z, reg_u in l_reg:
    cdl = BatchCDLfMRIFixedHRF(n_atoms=K, v=v, reg=reg_z, reg_u=reg_u,
                               n_iter=100, map_regu='l2', D_init='ssa',
                               random_state=None, n_jobs_u=2,
                               solver_d='only_u_adaptive', verbose=1)
    cdl.fit(noisy_X, nb_fit_try=3)
    pobj, times, uv_hat, z_hat = cdl.pobj_, cdl._times, cdl.uv_hat_, cdl.z_hat_

    z_hat = z_hat[0, :, :]
    Lz_hat = np.cumsum(z_hat, axis=-1)
    u_hat, v_hat = uv_hat[:, :P], uv_hat[0, P:]

    u_0_hat = u_hat[0, :]
    u_1_hat = u_hat[1, :]
    Lz_0_hat = Lz_hat[0, :].T
    Lz_1_hat = Lz_hat[1, :].T

    prod_scal_0 = np.dot(Lz_0_hat.flat, Lz[0, 0, :].T.flat)
    prod_scal_1 = np.dot(Lz_0_hat.flat, Lz[0, 1, :].T.flat)
    if prod_scal_0 < prod_scal_1:
        tmp = Lz_0_hat
        Lz_0_hat = Lz_1_hat
        Lz_1_hat = tmp
        tmp = u_0_hat
        u_0_hat = u_1_hat
        u_1_hat = tmp
    err_0_ = np.linalg.norm(u_0_hat - u_0) / np.linalg.norm(u_0)
    err_1_ = np.linalg.norm(u_1_hat - u_1) / np.linalg.norm(u_1)
    l_err.append(0.5 * (err_0_ + err_1_))
    l_res.append(dict(Lz_0_hat=Lz_0_hat, Lz_1_hat=Lz_1_hat, u_0_hat=u_0_hat,
                      u_1_hat=u_1_hat, pobj=pobj, times=times))

idx_best = np.argmin(l_err)
err_ = l_err[idx_best]
reg_z, reg_u = l_reg[idx_best]
print("Best regu: reg_z={:.3e}, reg_u={:.3e}".format(reg_z, reg_u))
res = list(l_res[idx_best].values())
Lz_0_hat, Lz_1_hat, u_0_hat, u_1_hat, pobj, times = res


###############################################################################
# check printings
# err_msg = "Est. map 1 have not the proper l1 norm"
# np.testing.assert_allclose(np.sum(np.abs(u_0)), np.sum(np.abs(u_0_hat)),
#                            err_msg=err_msg, rtol=1.0e-4)
# err_msg = "Est. map 2 have not the proper l1 norm"
# np.testing.assert_allclose(np.sum(np.abs(u_1)), np.sum(np.abs(u_1_hat)),
#                            err_msg=err_msg, rtol=1.0e-4)
# err_msg = "the fixed HRF was change during the estimation!"
# np.testing.assert_allclose(v.flat, v_hat.flat, err_msg=err_msg)

###############################################################################
# archiving results
res = dict(pobj=pobj, times=times, uv_hat=uv_hat, z_hat=z_hat, Lz_hat=Lz_hat,
           uv=uv, z=z, Lz=Lz, L=L)
filename = os.path.join(dirname, "results.pkl")
print("Pickling results under '{0}'".format(filename))
with open(filename, "wb") as pfile:
    pickle.dump(res, pfile)

###############################################################################
# plotting
# Lz
plt.figure("Temporal atoms", figsize=(12, 5))
plt.subplot(121)
plt.plot(Lz[0, 0, :].T, lw=2.0, label="True atom")
plt.plot(Lz_0_hat, lw=2.0, label="Est. atom")
x_0 = noisy_X[0, np.where(u_0 > 0)[0], :]
x_0 /= np.repeat(np.max(np.abs(x_0), axis=1)[:, None], X.shape[-1], 1)
t = np.arange(X.shape[-1])
mean_0 = np.mean(x_0, axis=0)
std_0 = np.std(x_0, axis=0)
borders_0 = (mean_0 - std_0, mean_0 + std_0)
plt.plot(mean_0, color='k', lw=0.5, label="Observed BOLD")
plt.fill_between(t, borders_0[0], borders_0[1], alpha=0.2, color='k')
plt.axhline(0.0, color='k', linewidth=0.5)
plt.xticks([0, T/2.0, T], fontsize=20)
plt.yticks([-1, 0, 1], fontsize=20)
plt.xlabel("Time [time-frames]", fontsize=20)
plt.legend(ncol=2, loc='lower center', fontsize=17, framealpha=0.3)
plt.title("First atom", fontsize=20)
plt.subplot(122)
plt.plot(Lz[0, 1, :].T, lw=2.0, label="True atom")
plt.plot(Lz_1_hat, lw=2.0, label="Est. atom")
x_1 = noisy_X[0, np.where(u_1 > 0)[0], :]
x_1 /= np.repeat(np.max(np.abs(x_1), axis=1)[:, None], X.shape[-1], 1)
mean_1 = np.mean(x_1, axis=0)
std_1 = np.std(x_1, axis=0)
borders_1 = (mean_1 - std_1, mean_1 + std_1)
plt.plot(mean_1, color='k', lw=0.5, label="Observed BOLD")
plt.fill_between(t, borders_1[0], borders_1[1], alpha=0.2, color='k')
plt.axhline(0.0, color='k', linewidth=0.5)
plt.xticks([0, T/2.0, T], fontsize=20)
plt.yticks([-1, 0, 1], fontsize=20)
plt.xlabel("Time [time-frames]", fontsize=20)
plt.legend(ncol=2, loc='lower center', fontsize=17, framealpha=0.3)
plt.title("Second atom", fontsize=20)
plt.tight_layout()
filename = "Lz.pdf"
filename = os.path.join(dirname, filename)
plt.savefig(filename, dpi=150)
subprocess.call("pdfcrop {}".format(filename), shell=True)
os.rename(filename.split('.')[0]+'-crop.pdf', filename)
print("Saving plot under '{0}'".format(filename))

# U
fig, axes = plt.subplots(nrows=1, ncols=4)
l_u = [u_0.reshape(p, p), u_0_hat.reshape(p, p),
       u_1.reshape(p, p), u_1_hat.reshape(p, p)]
l_max_u = [np.max(u) for u in l_u]
max_u = np.max(l_max_u)
amax_u = np.argmax(l_max_u)
l_name = ["True map 1", "Est. map 1", "True map 2", "Est. map 2"]
l_im = []
for ax, u, name in zip(axes.flat, l_u, l_name):
    l_im.append(ax.matshow(u))
    ax.set_title(name, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
fig.subplots_adjust(bottom=0.1, top=0.5, left=0.1, right=0.8,
                    wspace=0.3, hspace=0.2)
cbar_ax = fig.add_axes([0.83, 0.2, 0.02, 0.2])
cbar = fig.colorbar(l_im[amax_u], cax=cbar_ax)
cbar.set_ticks(np.linspace(0.0, max_u, 3))
filename = "U.pdf"
filename = os.path.join(dirname, filename)
plt.savefig(filename, dpi=150)
subprocess.call("pdfcrop {}".format(filename), shell=True)
os.rename(filename.split('.')[0]+'-crop.pdf', filename)
print("Saving plot under '{0}'".format(filename))

# pobj
plt.figure("Cost function (%)", figsize=(5, 5))
values = np.array(pobj)
values /= (100.0 * values[0])
plt.plot(values, lw=2.0)
plt.title("Evolution of global cost-function")
plt.xlabel('iterations')
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
