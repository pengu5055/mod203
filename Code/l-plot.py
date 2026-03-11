"""
Plot computed eigenvalues for Hydrogen scan over l values.
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

fig, ax = plt.subplots(1, 2, figsize=(12, 7))
cm = cmr.get_sub_cmap("cmr.tropical", 0.0, 0.8)
cm.set_bad(color="black")
norm = mpl.colors.LogNorm(vmin=np.nanmin(rel_errs), vmax=np.nanmax(rel_errs))

img = ax[0].imshow(rel_errs.T, aspect='auto', cmap=cm, norm=norm, origin='lower',
                extent=[l_vals[0], l_vals[-1], 1, max_eigenvalues], zorder=1)
ax[0].set_xlabel("Angular Momentum Quantum Number $l$")
ax[0].set_ylabel("Eigenvalue Index $j$")
ax[0].set_title("Relative Deviation of Numerical Values\nfrom Theoretical Energy Levels")
ax[0].grid(False)
cbar = fig.colorbar(img, ax=ax[0], label=r"$|\Delta E|/E_{\text{Theory}}$ from $E_j = -\frac{1}{2j^2}$ for $j = l + 1, l + 2, ...$")

# Add black rectangle patch for bad data legend
bad_patch = Patch(color='black', label='No Eigenvalue Found')
ax[0].legend(handles=[bad_patch], loc='upper right', frameon=True)

# Plot grid for easier visualization
ls_box = np.linspace(l_vals[0], l_vals[-1], len(l_vals)+1)
print(eigenvalues_array.T.shape)
hs_box = np.linspace(1, max_eigenvalues, max_eigenvalues+1)
for i in range(len(ls_box)):
    for j in range(len(hs_box)):
        x1 = np.array([ls_box[i], ls_box[i]])
        y1 = np.array([hs_box[j], hs_box[j-1]])
        x2 = np.array([ls_box[i-1], ls_box[i]])
        y2 = np.array([hs_box[j], hs_box[j]])

        flag = np.isnan(eigenvalues_array[i-1, j-1])
        if flag:
            grid_c = "white"
        else:
            grid_c = "black"

        ax[0].plot(x1, y1, color=grid_c, lw=0.5, zorder=2)
        ax[0].plot(x2, y2, color=grid_c, lw=0.5, zorder=2)
        ax[0].plot([ls_box[0], ls_box[0]], [hs_box[0], hs_box[-1]], color="k", lw=2.5, zorder=2)
        ax[0].plot([ls_box[-1], ls_box[-1]], [hs_box[0], hs_box[-1]], color="k", lw=2.5, zorder=2)
        ax[0].plot([ls_box[0], ls_box[-1]], [hs_box[0], hs_box[0]], color="k", lw=2, zorder=2)
        ax[0].plot([ls_box[0], ls_box[-1]], [hs_box[-1], hs_box[-1]], color="k", lw=2, zorder=2)

# Center ticks on integer l values and eigenvalue indices
ax[0].set_xticks(l_vals + 0.5 - np.array([(j-1)/len(l_vals) for j in range(1, len(l_vals)+1)]))
ax[0].set_xticklabels([f"{l}" for l in l_vals])
ax[0].set_yticks((np.arange(1, max_eigenvalues + 1) + 0.5) - np.array([(j-1)/max_eigenvalues for j in range(1, max_eigenvalues + 1)]))
ax[0].set_yticklabels([f"{j}" for j in range(1, max_eigenvalues + 1)])

plt.tight_layout()
plt.savefig(f"./Images/hydrogen_eigenvalues_l{l_min}_to_{l_max}_r{r_max}_g{int(np.log10(granularity))}.png", dpi=450)
plt.show()
