"""
Plot dispersion relation from k-scan results.
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

k_start = 0.8
k_end = 10.0
N = 5
x_max = 10.0
granularity = 100000
n_scan = 100

fn = f"./Data/fiber_dispersion_k{k_start}_to_{k_end}_N{N}_x{x_max}_g{int(np.log10(granularity))}.npz"

data = np.load(fn, allow_pickle=True)
k_vals = data["k_vals"]
eigenvalues = [data[f"k_{i}"] for i in range(len(k_vals))]

fig, ax = plt.subplots(figsize=(8, 6))
idx_max = max(len(evs) for evs in eigenvalues)
colors = cmr.take_cmap_colors("cmr.tropical", idx_max, cmap_range=(0.0, 0.8))

for i, (k_val, evs) in enumerate(zip(k_vals, eigenvalues)):
    for j, lam in enumerate(sorted(evs)):
        ax.scatter(k_val, lam, color=colors[j % len(colors)], s=20, label=f"Mode {j}" if i == 2 else "")
ax.set_xlabel('k')
ax.set_ylabel('$\lambda$')
ax.set_title('Dispersion Relation $\lambda(k)$ for Fiber Modes')
plt.legend(title="Modes", loc="upper left")
plt.tight_layout()
plt.show()
