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

l_min = 0
l_max = 9
r_max = 500.0
granularity = 100000

fn = f"./Data/hydrogen_eigenvalues_l{l_min}_to_{l_max}_r{r_max}_g{int(np.log10(granularity))}.npz"

data = np.load(fn, allow_pickle=True)
l_vals = data["l_vals"]
eigenvalues = [data[f"l_{l_val}"] for l_val in l_vals]

for l_val in l_vals:
    print(f"l={l_val}: Found {len(data[f'l_{l_val}'])} eigenvalues: {data[f'l_{l_val}']}")

# Now stack l values into (l, max_eigenvalues) array for plotting
max_eigenvalues = max(len(evs) for evs in eigenvalues)
eigenvalues_array = np.full((len(l_vals), max_eigenvalues), np.nan)  # Fill with NaN for missing values
for i, evs in enumerate(eigenvalues):
    eigenvalues_array[i, :len(evs)] = evs

rel_errs = np.full_like(eigenvalues_array, np.nan)

# Subtract theoretical values E_n = -1/(2*n^2), n_min = l + 1
for i, l_val in enumerate(l_vals):
    n_min = l_val + 1
    for j in range(max_eigenvalues):
        n = n_min + j
        theoretical_E = -1/(2*n**2)
        if not np.isnan(eigenvalues_array[i, j]):
            rel_errs[i, j] = np.abs(eigenvalues_array[i, j] - theoretical_E) / np.abs(theoretical_E)

# Take abs deviations
eigenvalues_array = np.abs(eigenvalues_array)

# Get wavefunctions for j = l + 1
states = []
for i, l_val in enumerate(l_vals):
    l_eig = eigenvalues[i]
    x_ranges, wf_shoot = get_hydrogen_wavefunctions(l_eig, l=l_val, multiple=7*(i/2+1), r_min=1e-2)

    states.append((x_ranges[i], wf_shoot[i]))

states = states[:5]

# Plotting
colors = cmr.take_cmap_colors("cmr.tropical", len(eigenvalues), cmap_range=(0.0, 0.8))
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax = [ax]

for i, (x, wave) in enumerate(states):
    analytic = hydrogen_analytic(x, n=i+1, l=i)
    ax[0].plot(x, wave, color=colors[i])


ax[0].set_title("Radial Wavefunctions")
ax[0].set_xlabel("r")
ax[0].set_ylabel("R(r)")
ax[0].legend()
plt.tight_layout()
plt.show()
