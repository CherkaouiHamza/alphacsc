import numpy as np
import matplotlib.pyplot as plt

from alphacsc.simulate import hrf
from alphacsc import learn_d_z_multi
from alphacsc.utils.dictionary import get_uv
from alphacsc.loss_and_gradient import construct_X_multi
from alphacsc.init_dict import init_dictionary


###############################################################################
# define data
T = 110  # BOLD time frames
p = 6
P = p*p  # number of voxels
L = 60   # HRF time frames
K = 2    # number of temporal atoms
N = 1    # n subject

assert T == 110
assert K == 2
assert N == 1

random_seed = None
rng = np.random.RandomState(random_seed)

# define z (blocks)
blocks_0 = np.array(([0.0] * 10 + [1.0] * 10) * 5 + [0.0] * 10)
blocks_1 = np.array(([0.0] * 10 + [1.0] * 10 + [0.0] * 30) * 2 + [0.0] * 10)
Lz = np.vstack([blocks_0, blocks_1])[None]
assert Lz.shape == (N, K, T)

# add noise on z
sigma = 1.0e-1
noise = sigma * rng.randn(*Lz.shape)
Lz += noise

# define v (HRF)
v = hrf(L).reshape(L, 1)
v /= np.linalg.norm(v)

# define u (spatial map)
u_0 = np.vstack([np.ones((int(p/2), p), np.float),
                 np.zeros((int(p/2), p), np.float)]).reshape(P, 1)
u_1 = np.vstack([np.zeros((int(p/2), p), np.float),
                 np.ones((int(p/2), p), np.float)]).reshape(P, 1)

# normalized each u_i
u_0 /= np.linalg.norm(u_0)
u_1 /= np.linalg.norm(u_1)

uv_0 = u_0.dot(v.T)
uv_1 = u_1.dot(v.T)

# define D (spatio-temporal map)
D = np.stack([uv_0, uv_1])
uv = get_uv(D)
assert uv.shape == (K, P+L)
assert D.shape == (K, P, L)

# define X
X = construct_X_multi(Lz, D, n_channels=P)

###############################################################################
# estimation of d an z
reg = 1.0
n_iter = 50
nb_runs = 5
results = []

for ii in range(1, nb_runs+1):
    print("Run: {}/{}".format(ii, nb_runs))

    uv_init = init_dictionary(X, K, L, uv_constraint='separate',
                              rank1=True, window=False, D_init='ssa',
                              D_init_params=dict(), random_state=random_seed)
    uv_init[:, P:] = np.repeat(v[None, :], K, axis=0).reshape(K, L)

    pobj, _, d_hat, z_hat, _ = learn_d_z_multi(
        X, K, L, reg=reg, lmbd_max='scaled', n_iter=n_iter,
        D_init=uv_init, solver_z='fista', solver_d='only_u_adaptive',
        positivity=False, random_state=rng, loss_params=dict(block=True),
        raise_on_increase=True, n_jobs=1, verbose=1)

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
plt.plot(Lz[0, :, :].T, lw=2.0)
plt.title("True blocks coding signals (Lz)")
plt.grid()
plt.subplot(212)
plt.plot(Lz_hat[0, :, :].T, lw=2.0)
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

# v
plt.figure("HRF", figsize=(5, 4))
plt.subplot(121)
plt.plot(v, lw=2.0)
plt.xlabel("time frames")
plt.title("True HRF")
plt.grid()
plt.subplot(122)
plt.plot(v_hat, lw=2.0)
plt.xlabel("time frames")
plt.title("Est. HRF")
plt.grid()
plt.tight_layout()

# pobj
plt.figure("Cost function (%)", figsize=(4, 4))
values = np.array(pobj)
values /= (100.0 * values[0])
plt.semilogx(values, lw=2.0)
plt.title("Evolution of global cost-function")
plt.xlabel('iter')
plt.grid()

plt.show()
