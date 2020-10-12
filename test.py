import numpy as np
from triqs_tprf.tight_binding import TBLattice
import triqs_tprf as trpf
from pytriqs.gf import *
from triqs_tprf.lattice import lattice_dyson_g0_wk
import matplotlib.pyplot as plt
t = 1.0
H = TBLattice(
    units=[(1, 0, 0), (0, 1, 0)],
    hopping={
        # nearest neighbour hopping -t
        (0, +1): -t * np.eye(2),
        (0, -1): -t * np.eye(2),
        (+1, 0): -t * np.eye(2),
        (-1, 0): -t * np.eye(2),
    },
    orbital_positions=[(0, 0, 0)] * 2,
    orbital_names=['up', 'do'],
)
e_k = H.on_mesh_brillouin_zone(n_k=(32, 32, 1))

beta = 50
n_iw = 130
mu = -5
imesh = MeshImFreq(beta, 'Fermion', n_iw)
g = lattice_dyson_g0_wk(mu, e_k, imesh)

gm = GfImFreq(mesh=imesh, data=g[(0, 0)].data[:, 0].reshape(-1, 1, 1), beta=50)
g_pade = GfReFreq(window=(-2, 2), n_points=200, target_shape=[1, 1])
g_pade.set_from_pade(gm)
print(g_pade.data.shape)