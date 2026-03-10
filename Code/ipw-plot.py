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

step_size = 1/200000
upper = 500

ts = time()
ipw_x, ipw_psi, ipw_ana, ipw_E = potwell(psi_init, upper, step_size)
te = time()
print(f"Time taken: {te - ts:.2f} seconds")
print(f"Found energy levels: {ipw_E}")

fig, ax = plt.subplots(figsize=(8, 5))
colors = cmr.take_cmap_colors("cmr.tropical", len(ipw_E), cmap_range=(0.0, 0.8))
for i, psi in enumerate(ipw_psi):
    ax.plot(ipw_x, psi, label=f"$E_{i}={ipw_E[i]:.2f}$", color=colors[i])
ax.set_xlabel('x')
ax.set_ylabel('$\psi(x)$')
ax.legend(ncols=2, loc="lower left", fontsize=10)
plt.suptitle(f"Particle in a Box aka Infinite Potential Well")
par_str = f"Evaluated with PEFRL method, $\log_{{10}}(g)={int(np.log10(1/step_size))}$, $E_{{max}}={upper}$"
plt.title(par_str, fontsize=10)
plt.tight_layout()
plt.subplots_adjust(top=0.87)
plt.savefig(f"./Images/particle_in_box_infinite_upper{upper}_step{step_size}.png", dpi=450)
plt.show()

