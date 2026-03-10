"""
Plot with legacy ancient code for particle in a box.
Source: https://forge.ephemera.zip/pengu5055/mfp08/src/branch/main/src_shoot.py
"""
import numpy as np
from src import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
import os
from time import time
from src_shoot import potwell

mpl.style.use("./vrm.mplstyle")
mpl.use("qtagg")

psi_0 = 0.0
psi_prime_0 = 1.0
psi_init = np.array([psi_0, psi_prime_0])

step_size = 1/10000
upper = 500

ts = time()
ipw_x, ipw_psi, ipw_ana, ipw_E = potwell(psi_init, upper, step_size)
te = time()
print(f"Time taken: {te - ts:.2f} seconds")
print(f"Found energy levels: {ipw_E}")

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
colors = cmr.take_cmap_colors("cmr.tropical", len(ipw_E), cmap_range=(0.0, 0.8))
for i, psi in enumerate(ipw_psi):
    ax[0].plot(ipw_x, psi, label=f"$E_{i}={ipw_E[i]:.2f}$", color=colors[i])
ax[0].set_xlabel('x')
ax[0].set_ylabel('$\psi(x)$')
ax[0].set_title("Numerically Evaluated Wavefunctions")

for i, (psi, ana) in enumerate(zip(ipw_psi, ipw_ana)):
    ax[1].plot(ipw_x, aerr(ana, psi), label=f"$E_{i}={ipw_E[i]:.2f}$", color=colors[i])
ax[1].set_xlabel('x')
ax[1].set_ylabel('Absolute Error $| \psi_{ana} - \psi_{num} |$')
ax[1].set_yscale('log')
ax[1].set_title("Abs. Err. Compared to Analytical Solution")

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False, prop={"size": 10, "weight": "medium"}, 
           title="Eigenenergies", title_fontproperties={"size": 12, "weight": "bold"})

plt.suptitle(f"Eigen-functions and Energies for an Infinite Potential Well")
plt.tight_layout()
par_str = f"Evaluated via Shooting with PEFRL method, $\log_{{10}}(g)={int(np.log10(1/step_size))}$, $E_{{max}}={upper}$"
plt.subplots_adjust(top=0.825, bottom=0.15, right=0.97)
plt.text(0.5, 0.93, par_str, ha="center", va="top", transform=plt.gcf().transFigure, fontsize=10, weight="medium")
plt.savefig(f"./Images/particle_in_box_infinite_upper{upper}_step{step_size}.png", dpi=450)
plt.show()
