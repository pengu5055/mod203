"""
Test WIP Code.
"""
import numpy as np
from src import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
from time import time

ts = time()

r_range = np.linspace(1e-4, 10, 100000)

k_func = lambda r, E: k_hydrogen(r, E, l=0)
seed_func = lambda r, E: seed_hydrogen(r, E, l=0)

eigenvalues = find_eigenvalues(r_range, k_func, seed_func, shoot_par_range=(-0.6, -0.01))

print(f"Found {len(eigenvalues)} as: ")
for i, (E_left, E_right) in enumerate(eigenvalues):
    print(f"  E_{i} in [{E_left:.4f}, {E_right:.4f}]")

te = time()
print(f"Time taken: {te - ts:.2f} seconds")

# Plotting
colors = cmr.take_cmap_colors("cmr.tropical", 8, cmap_range=(0.0, 0.8))
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

for i in range(3):
    wave = get_wavefunction(r_range, k_func, seed_func, shoot_par=np.mean(eigenvalues[i]))
    ax[0].plot(r_range, wave, label=f"n={i+1}", color=colors[i])
    ax[1].plot(r_range, k_func(r_range, np.mean(eigenvalues[i])), label=f"n={i+1}", color=colors[i])

ax[0].set_title("Radial Wavefunctions")
ax[0].set_xlabel("r")
ax[0].set_ylabel("u(r)")
ax[0].legend()

ax[1].set_title("k(r) for Hydrogen")
ax[1].set_xlabel("r")
ax[1].set_ylabel("k(r)")
ax[1].legend()

plt.tight_layout()
plt.show()
