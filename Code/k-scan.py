"""
Scan over k_vals for fiber problem to find dispersion relation.
"""
import numpy as np
from src import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
import os
from time import time

mpl.style.use("./vrm.mplstyle")
mpl.use("tkagg")

ts = time()
CACHE = False
x_max = 10
granularity = 10000
k_vals = np.linspace(0.8, 10, 200)
lam_range = lambda a: (1.0 * a + 1e-3, 2.0 * a - 1e-3)
x_range = np.linspace(1e-3, x_max, granularity)

core_boundary_idx = np.searchsorted(x_range, 1.0)

k_func = lambda r, lam: k_fiber(r, lam, k=k_val)
seed_out = lambda r, lam: seed_fiber(r, lam)
seed_in = lambda r, lam, idx: seed_fiber_inward(r, lam, k_val, idx)

fn = f"./Data/hh_fiber_x{x_max}_ks{k_vals[0]}_ke{k_vals[-1]}_g{int(np.log10(granularity))}.npy"

if not os.path.exists(fn) or not CACHE:
    eigenvalues = find_eigenvalues(x_range, k_func, (seed_out, seed_in),
                               shoot_par_range=lam_range,
                               shoot_func=shoot_midpoint,
                               match_idx=core_boundary_idx,
                               inward_buffer=5.0, n_scan=100, negate_k=True)

    print(f"Found {len(eigenvalues)} eigenvalues up to x={x_max} ")
    for i, e in enumerate(eigenvalues):
        print(f"$\lambda$_{i}: {e:.6f}")
    np.save(fn, eigenvalues)
else:
    eigenvalues = np.load(fn)

te = time()
print(f"Time taken: {te - ts:.2f} seconds")
