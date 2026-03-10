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
from src_shoot import finpotwell

mpl.style.use("./vrm.mplstyle")
mpl.use("tkagg")

CACHE = True
psi_0 = 0.0
psi_prime_0 = 1.0
psi_init = np.array([psi_0, psi_prime_0])
depth = 100
scale = 50
step_size = 1/2000
upper = 100 
x_max = 10.0
granularity_step = 1
x_range = np.arange(-x_max, x_max + step_size, step_size)
fn = f"./Data/particle_in_box_finite_upper{upper}_depth{depth}_step{step_size}_x{x_max}_gs{granularity_step}.npz"

ts = time()
if not os.path.exists(fn) or not CACHE:
    fpw_x, fpw_psi, fpw_ana, fpw_E, fpw_V = finpotwell(psi_init, upper, depth, step_size, step=granularity_step)
    np.savez(fn, fpw_x=fpw_x, fpw_psi=fpw_psi, fpw_ana=fpw_ana, fpw_E=fpw_E, fpw_V=fpw_V)
else:
    data = np.load(fn, allow_pickle=True)
    fpw_x = data["fpw_x"]
    fpw_psi = data["fpw_psi"]
    fpw_ana = data["fpw_ana"]
    fpw_E = data["fpw_E"]
    fpw_V = data["fpw_V"]

te = time()
print(f"Time taken: {te - ts:.2f} seconds")
print(f"Found energy levels: {fpw_E}")

# Truncate all to keep only 6 eigenvalues
fpw_psi = fpw_psi[:6]
fpw_ana = fpw_ana[:6]
fpw_E = fpw_E[:6]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
colors = cmr.take_cmap_colors("cmr.tropical", len(fpw_E), cmap_range=(0.0, 0.8))
for i, psi in enumerate(fpw_psi):
    ax[0].plot(fpw_x, scale*psi, color=colors[i], zorder=3)

ax[0].plot(fpw_x, fpw_V, label="Potential", color='k', lw=1)
ax[0].set_xlabel('x')
ax[0].set_ylabel('$\psi(x)$')
ax[0].set_title("Numerically Evaluated Wavefunctions")
ax[0].set_xlim(-2, 2)
ax[0].set_ylim(-100, 125)
ax[0].legend(frameon=False, loc="lower right")

for i, (psi, ana) in enumerate(zip(fpw_psi, fpw_ana)):
    # ax[1].plot(fpw_x, aerr(ana, psi), label=f"$E_{i}={fpw_E[i]:.2f}$", color=colors[i])
    ax[1].plot(fpw_x, ana, label=f"$E_{i}={fpw_E[i]:.2f}$", color=colors[i], zorder=3)
ax[1].set_xlabel('x')
ax[1].set_ylabel('Absolute Error $| \psi_{ana} - \psi_{num} |$')
ax[1].set_title("Abs. Err. Compared to Analytical Solution")
ax[1].set_xlim(-2, 2)
# ax[1].set_ylim(1e-9, 1e1)
# ax[1].set_yscale('log')


handles, labels = ax[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False, prop={"size": 10, "weight": "medium"}, 
           title="Eigenenergies", title_fontproperties={"size": 12, "weight": "bold"})

plt.suptitle(f"Eigen-functions and Energies for a Finite Potential Well of Depth {depth} a.u.")
plt.tight_layout()
par_str = f"Evaluated via Shooting with PEFRL method, $\log_{{10}}(g)={int(np.log10(1/step_size))}$, $E_{{max}}={upper}$, $d={depth}$, $x_{{max}}={x_max}$, $A_{{scale}}={scale}$"
plt.subplots_adjust(top=0.825, bottom=0.15, right=0.97)
plt.text(0.5, 0.93, par_str, ha="center", va="top", transform=plt.gcf().transFigure, fontsize=10, weight="medium")
plt.savefig(f"./Images/particle_in_box_finite_upper{upper}_depth{depth}_step{step_size}.png", dpi=450)
plt.show()
