"""
Plot fiber wavefunctions for 4 different k values, 
"""
import numpy as np
from src import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
import os
from time import time

mpl.style.use("./vrm.mplstyle")
mpl.use("qtagg")

ts = time()
CACHE = True
x_max = 5
granularity = 100000
k_vals = [1, 2, 5, 10]
k_vals_str = '-'.join(str(k) for k in k_vals)
fn = f"./Data/hh_fiber_x{x_max}_k{k_vals_str}_g{int(np.log10(granularity))}.npz"

eigenvalues_dict = {}
if not os.path.exists(fn) or not CACHE:
    for k_val in k_vals:
        lam_range = (1.0 * k_val + 1e-3, 2.0 * k_val - 1e-3)
        x_range = np.linspace(0.1, x_max, granularity)
        core_boundary_idx = np.searchsorted(x_range, 1.0)
    
        k_func = lambda r, lam: k_fiber(r, lam, k=k_val)
        seed_out = lambda r, lam: seed_fiber(r, lam, k=k_val)
        seed_in = lambda r, lam, idx: seed_fiber_inward(r, lam, k_val, idx)
    
        eigenvalues = find_eigenvalues(x_range, k_func, (seed_out, seed_in),
                                shoot_par_range=lam_range,
                                shoot_func=shoot_midpoint,
                                match_idx=core_boundary_idx,
                                inward_buffer=5.0, n_scan=100, negate_k=True)
        eigenvalues_dict[f"k_{k_val}"] = eigenvalues

        print(f"Found {len(eigenvalues)} eigenvalues up to x={x_max} ")
        for i, e in enumerate(eigenvalues):
            print(f"$k={k_val}$, $\lambda$_{i}: {e:.6f}")
        
    print(f"Completed! Storing keys: {list(eigenvalues_dict.keys())}")
    np.savez(fn, **eigenvalues_dict, k_vals=k_vals)
else:
    data = np.load(fn, allow_pickle=True)
    k_vals = data["k_vals"]
    eigenvalues_dict = {f"k_{k_val}": data[f"k_{k_val}"] for k_val in k_vals}

te = time()
print(f"Time taken: {te - ts:.2f} seconds")

fig, ax = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
ax = ax.flatten()

for i, k_val in enumerate(k_vals):
    eigenvalues = eigenvalues_dict[f"k_{k_val}"]
    colors = cmr.take_cmap_colors("cmr.tropical", len(eigenvalues), cmap_range=(0.0, 0.8))

    xs, wfs = get_fiber_wavefunctions(eigenvalues, k_val, multiple=5, negate_k=False, inward_buffer=1000.0,
                                      mode="midpoint", x_min=1e-3, n_eval=10000, core_idx=10)
    
    for j, (x, wave) in enumerate(zip(xs, wfs)):
        ax[i].plot(x, wave, label=f"$\lambda_{j} = {eigenvalues[j]:.3f}$", color=colors[j], zorder=3)

    ax[i].axvline(x=1.0, color='k', linestyle='--', alpha=1.0)
    ax[i].text(1.1, -1.3, "Core Boundary", rotation=90, va="bottom", 
               ha="left", fontsize=8, color='k')
    ax[i].legend()
    ax[i].set_title(f"k={k_val}")
    ax[i].set_xlim(0, 1.5*x_max)


ax[2].set_xlabel('x')
ax[3].set_xlabel('x')
ax[0].set_ylabel('R(x)')
ax[2].set_ylabel('R(x)')

plt.suptitle(f"Fiber Modes for Different k Values")
parstr = f"Eigenvalues Evaluated with: $x_{{max}}={x_max}$, $N_{{scan}}=100$, $\log_{{10}}(g)={int(np.log10(granularity))}$"
wf_eval_str = f"Functions Evaluated w/: multiple=5, inward_buffer=1000.0, mode='midpoint',\n           core_idx=10, x_min=1e-3, n_eval=10000"
plt.tight_layout()
plt.subplots_adjust(top=0.865)
plt.text(0.5, 0.95, parstr, ha="center", va="top", transform=plt.gcf().transFigure, fontsize=10, weight="medium")
plt.text(0.5, 0.93, wf_eval_str, ha="center", va="top", transform=plt.gcf().transFigure, fontsize=8)
plt.savefig(f"./Images/hh_fiber_modes_k{k_vals_str}_x{x_max}_g{int(np.log10(granularity))}.png", dpi=450)
plt.show()

