"""
Test WIP Code.
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
CACHE = True
r_max = 100
granularity = 10000
r_range = np.linspace(1e-3, r_max, granularity)

k_func = lambda r, E: k_hydrogen(r, E, l=0)
seed_func_out = lambda r, E: seed_hydrogen(r, E, l=0)
seed_func_in = lambda r, E, idx: seed_hydrogen_inward(r, E, l=0, start_idx=idx)

fn = f"./Data/eigenvalues_r{r_max}_g{int(np.log10(granularity))}.npy"

if not os.path.exists(fn) or not CACHE:
    eigenvalues = find_eigenvalues(r_range, k_func, seed_func_out, 
                                   shoot_par_range=(-0.6, -0.01), 
                                   shoot_func=shoot,
                                   n_scan=1000,
                                   renorm_every=10)

    print(f"Found {len(eigenvalues)} eigenvalues up to r={r_max} ")
    for i, e in enumerate(eigenvalues):
        print(f"E_{i}: {e:.6f}")
    np.save(fn, eigenvalues)
else:
    eigenvalues = np.load(fn)

te = time()
print(f"Time taken: {te - ts:.2f} seconds")

# Plotting
colors = cmr.take_cmap_colors("cmr.tropical", len(eigenvalues), cmap_range=(0.0, 0.8))
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

xs, wfs = get_wavefunctions(eigenvalues, 0, idx=[0, 1, 2, 3, 4], multiple=[10, 5, 5, 5, 5])

for i, (x, wave) in enumerate(zip(xs, wfs)):
    print(f"Plotting n={i+1} with E={eigenvalues[i]:.6f}")
    analytic = hydrogen_analytic(x, n=i+1, l=0)
    ax[0].plot(x, wave, label=f"n={i+1}", color=colors[i])
    ax[0].plot(x, analytic, label=f"Analytic n={i+1}", linestyle="--", color=colors[i])

    ax[1].plot(x, aerr(analytic, wave), label=f"n={i+1}", color=colors[i])

ax[0].set_title("Radial Wavefunctions")
ax[0].set_xlabel("r")
ax[0].set_ylabel("u(r)")
ax[0].legend()

ax[1].set_title("Absolute Error")
ax[1].set_xlabel("r")
ax[1].set_ylabel("k(r)")
ax[1].legend()
ax[1].set_yscale("log")

plt.tight_layout()
plt.show()
