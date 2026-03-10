"""
Test WIP Code part 2.
"""
import numpy as np
from src import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
import os
from time import time

mpl.style.use("./vrm.mplstyle")
mpl.use("qtagg")

ts = time()
CACHE = True
x_max = 20
granularity = 100000
k_val = 2
lam_range = (1.0 * k_val + 1e-3, 2.0 * k_val - 1e-3)
x_range = np.linspace(0.1, x_max, granularity)



core_boundary_idx = np.searchsorted(x_range, 1.0)

k_func = lambda r, lam: k_fiber(r, lam, k=k_val)
seed_out = lambda r, lam: seed_fiber(r, lam, k=k_val)
seed_in = lambda r, lam, idx: seed_fiber_inward(r, lam, k_val, idx)

fn = f"./Data/hh_fiber_x{x_max}_k{k_val}_g{int(np.log10(granularity))}.npy"

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

fig, ax = plt.subplots(figsize=(8, 5))
colors = cmr.take_cmap_colors("cmr.tropical", len(eigenvalues), cmap_range=(0.0, 0.8))

xs, wfs = get_fiber_wavefunctions(eigenvalues, k_val, multiple=5, negate_k=False, inward_buffer=25.0,
                                  mode="midpoint", x_min=1e-3, n_eval=10000)
for i, (x, wave) in enumerate(zip(xs, wfs)):
    ax.plot(x, wave, label=f"$\lambda_{i}$", color=colors[i])

ax.axvline(x=1, color='k', linestyle='--', alpha=0.5, label='core boundary')
ax.set_xlabel('x')
ax.set_ylabel('R(x)')
ax.set_title(f'Fiber modes k={k_val}')
ax.legend()
plt.tight_layout()
plt.show()
