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
abs_errs = np.full_like(eigenvalues_array, np.nan)

# Subtract theoretical values E_n = -1/(2*n^2), n_min = l + 1
for i, r_val in enumerate(r_scan_values):
    n_min = l_val + 1
    for j in range(max_eigenvalues):
        n = n_min + j
        theoretical_E = -1/(2*n**2)
        if not np.isnan(eigenvalues_array[i, j]):
            rel_errs[i, j] = np.abs(eigenvalues_array[i, j] - theoretical_E) / np.abs(theoretical_E)
            abs_errs[i, j] = np.abs(eigenvalues_array[i, j] - theoretical_E)

print(eigenvalues_array.shape, rel_errs.shape)
num_eigvals = max_eigenvalues
fig, ax = plt.subplots(1, 1, figsize=(8, 7))
ax = [ax]
colors = cmr.take_cmap_colors("cmr.tropical", num_eigvals, cmap_range=(0.0, 0.8))
cm = cmr.get_sub_cmap("cmr.tropical", 0.0, 0.8)
cm.set_bad(color="black")

eigenvalues_array = np.abs(eigenvalues_array)

for i in range(num_eigvals):
    r_norm = r_scan_values / (i + 1)**2
    ax[0].plot(r_norm, rel_errs[:, i], label=f"$E_{{{i+1}}}$", color=colors[i])

ax[0].set_xlabel("$r_\\text{max} / n^2$")
ax[0].set_ylabel("$|E_{ana} - E_{num}|$")
ax[0].set_title("Absolute Deviation from Analytical Value")
ax[0].set_yscale("log")
ax[0].set_xlim(0, 15)
ax[0].legend(ncols=2, fontsize=8)

plt.suptitle("Convergence of Eigenvalues with Increasing\n$r_\\text{max}$ for Hydrogen Atom ($l=0$)")
plt.tight_layout()
plt.subplots_adjust(top=0.805)
par_str = f"Evaluated for $r_{{max}}\in[{r_scan_start}, {r_scan_end}]$, $E\in[-0.6, -0.0001]$, $\log_{{10}}(g)={int(np.log10(granularity))}$, $n_{{scan}}=5000$"
plt.text(0.5, 0.885, par_str, ha='center', va='center', transform=plt.gcf().transFigure, fontsize=10, weight="medium")
plt.savefig(f"./Images/rmax_scan_l{l_val}_rscan{r_scan_start}_to_{r_scan_end}_g{int(np.log10(granularity))}.png", dpi=450)
plt.show()
