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

# Evaluate Avg. Abs. Dev. for Wavefunction via Shooting vs. Analytical
wf_abs_devs = np.full((len(l_vals), max_eigenvalues), np.nan)
for i, l_val in enumerate(l_vals):
    l_eig = eigenvalues[i]
    x_ranges, wf_shoot = get_hydrogen_wavefunctions(l_eig, l=l_val)
    wf_ana = np.array([hydrogen_analytic(x_ranges[i], n=l_val + 1 + j, l=l_val) for j in range(len(l_eig))])

    wf_abs_devs[i, :len(l_eig)] = np.array([np.mean(np.abs(wf_shoot[j] - wf_ana[j])) for j in range(len(l_eig))])

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



for a in ax:
    # Center ticks on integer l values and eigenvalue indices
    a.set_xticks(l_vals + 0.5 - np.array([(j-1)/len(l_vals) for j in range(1, len(l_vals)+1)]))
    a.set_xticklabels([f"{l}" for l in l_vals])
    a.set_yticks((np.arange(1, max_eigenvalues + 1) + 0.5) - np.array([(j-1)/max_eigenvalues for j in range(1, max_eigenvalues + 1)]))
    a.set_yticklabels([f"{j}" for j in range(1, max_eigenvalues + 1)])

    # Add black rectangle patch for bad data legend
    bad_patch = Patch(color='black', label='No Eigenvalue Found')
    a.legend(handles=[bad_patch], loc='upper right', frameon=True)
    
    # Plot grid for easier visualization
    ls_box = np.linspace(l_vals[0], l_vals[-1], len(l_vals)+1)
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

            a.plot(x1, y1, color=grid_c, lw=0.5, zorder=2)
            a.plot(x2, y2, color=grid_c, lw=0.5, zorder=2)
            a.plot([ls_box[0], ls_box[0]], [hs_box[0], hs_box[-1]], color="k", lw=2.5, zorder=2)
            a.plot([ls_box[-1], ls_box[-1]], [hs_box[0], hs_box[-1]], color="k", lw=2.5, zorder=2)
            a.plot([ls_box[0], ls_box[-1]], [hs_box[0], hs_box[0]], color="k", lw=2, zorder=2)
            a.plot([ls_box[0], ls_box[-1]], [hs_box[-1], hs_box[-1]], color="k", lw=2, zorder=2)

norm2 = mpl.colors.LogNorm(vmin=np.nanmin(wf_abs_devs), vmax=np.nanmax(wf_abs_devs))
img2 = ax[1].imshow(wf_abs_devs.T, aspect='auto', cmap=cm, norm=norm2, origin='lower',
                extent=[l_vals[0], l_vals[-1], 1, max_eigenvalues], zorder=1)
ax[1].set_xlabel("Angular Momentum Quantum Number $l$")
ax[1].set_ylabel("Eigenvalue Index $j$")
ax[1].set_title("Average Absolute Deviation of\nNumerical Wavefunction from Analytical Solution")
ax[1].grid(False)
cbar2 = fig.colorbar(img2, ax=ax[1], label=r"$\overline{|\psi_{\text{num}} - \psi_{\text{ana}}|}$")
plt.suptitle(f"Accuracy of Numerical Hydrogen Eigenvalues and Wavefunctions Compared to Theory")
plt.tight_layout()
plt.subplots_adjust(top=0.82)
par_str = f"Evaluated for $l\in[{l_min}, {l_max}]$, $E\in[-0.6, -0.00001]$, $r_{{max}}={r_max}$, $\log_{{10}}(g)={int(np.log10(granularity))}$, $n_{{scan}}=5000$"
plt.text(0.5, 0.93, par_str, ha='center', va='center', transform=plt.gcf().transFigure, fontsize=10, weight="medium")
plt.savefig(f"./Images/hydrogen_eigenvalues_l{l_min}_to_{l_max}_r{r_max}_g{int(np.log10(granularity))}.png", dpi=450)
plt.show()
