"""
Plot results of r_max scan for Hydrogen atom. 
"""
import numpy as np
from src import *
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib as mpl
import cmasher as cmr
import os
from time import time

mpl.style.use("./vrm.mplstyle")
mpl.use("tkagg")

r_scan_start = 5.0
r_scan_end = 500.0
l_val = 0
r_max = 500.0
granularity = 1000

fn = f"./Data/rmax_scan_l{l_val}_rscan{r_scan_start}_to_{r_scan_end}_g{int(np.log10(granularity))}.npz"

data = np.load(fn, allow_pickle=True)
r_scan_values = data["r_scan_values"]
eigenvalues = [data[f"r_{i}"] for i, r_val in enumerate(r_scan_values)]

# Now stack r_scan_values into (r_val, max_eigenvalues) array for plotting
max_eigenvalues = max(len(evs) for evs in eigenvalues)
eigenvalues_array = np.full((len(r_scan_values), max_eigenvalues), np.nan)  # Fill with NaN for missing values
for i, evs in enumerate(eigenvalues):
    eigenvalues_array[i, :len(evs)] = evs

# Print lines 
for i in range(eigenvalues_array.shape[0]):
    r_val = r_scan_values[i]
    eigs = eigenvalues_array[i, :np.sum(~np.isnan(eigenvalues_array[i]))]
    print(f"r_max={r_val:.2f}: Found {len(eigs)} eigenvalues: {eigs}")

rel_errs = np.full_like(eigenvalues_array, np.nan)

# Subtract theoretical values E_n = -1/(2*n^2), n_min = l + 1
for i, r_val in enumerate(r_scan_values):
    n_min = l_val + 1
    for j in range(max_eigenvalues):
        n = n_min + j
        theoretical_E = -1/(2*n**2)
        if not np.isnan(eigenvalues_array[i, j]):
            rel_errs[i, j] = np.abs(eigenvalues_array[i, j] - theoretical_E) / np.abs(theoretical_E)

print(eigenvalues_array.shape, rel_errs.shape)
num_eigvals = 6
fig, ax = plt.subplots(1, 2, figsize=(12, 7))
colors = cmr.take_cmap_colors("cmr.tropical", num_eigvals, cmap_range=(0.0, 0.8))

for i in range(6):
    ax[0].plot(r_scan_values, eigenvalues_array[:, i], label=f"Eigenvalue {i+1}", color=colors[i])
    ax[1].plot(r_scan_values, rel_errs[:, i], label=f"Eigenvalue {i+1}", color=colors[i])

ax[0].set_xlabel("r_max")
ax[0].set_ylabel("Abs. Eigenvalue")
ax[0].set_title("Abs. Eigenvalues vs. r_max")

ax[1].set_xlabel("r_max")
ax[1].set_ylabel("Relative Error")
ax[1].set_title("Relative Error vs. r_max")
ax[1].set_yscale("log")

plt.tight_layout()
plt.show()
