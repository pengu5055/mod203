"""
Plot computed eigenvalues for Hydrogen scan over l values.
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

fig, ax = plt.subplots(figsize=(8, 6))
cm = cmr.get_sub_cmap("cmr.chroma", 0.0, 0.75)
cm.set_bad(color="gray")
norm = mpl.colors.Normalize(vmin=np.nanmin(eigenvalues_array), vmax=np.nanmax(eigenvalues_array))

img = ax.imshow(eigenvalues_array.T, aspect='auto', cmap=cm, norm=norm, origin='lower',
                extent=[l_vals[0], l_vals[-1], 1, max_eigenvalues], zorder=3)
ax.set_xlabel("Angular Momentum Quantum Number $l$")
ax.set_ylabel("Eigenvalue Index")
ax.set_title("Eigenvalues for Hydrogen Atom as a Function of $l$")
ax.grid(False)
cbar = fig.colorbar(img, ax=ax)
cbar.set_label("Eigenvalue")
plt.tight_layout()
plt.savefig(f"./Images/hydrogen_eigenvalues_l{l_min}_to_{l_max}_r{r_max}_g{int(np.log10(granularity))}.png", dpi=450)
plt.show()
