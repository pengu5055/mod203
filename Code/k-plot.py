"""
Plot dispersion relation from k-scan results.
"""
import numpy as np
from src import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr

mpl.style.use("./vrm.mplstyle")
mpl.use("tkagg")

k_start = 0.8
k_end = 10.0
N = 1000
x_max = 10.0
granularity = 100000
n_scan = 500

fn = f"./Data/fiber_dispersion_k{k_start}_to_{k_end}_N{N}_x{x_max}_g{int(np.log10(granularity))}.npz"

data = np.load(fn, allow_pickle=True)
k_vals = data["k_vals"]
eigenvalues = [data[f"k_{i}"] for i in range(len(k_vals))]

fig, ax = plt.subplots(figsize=(8, 6))
idx_max = max(len(evs) for evs in eigenvalues)
colors = cmr.take_cmap_colors("cmr.tropical", idx_max, cmap_range=(0.0, 0.8))

for i, (k_val, evs) in enumerate(zip(k_vals, eigenvalues)):
    for j, lam in enumerate(sorted(evs)):
        ax.scatter(k_val, lam, color=colors[j % len(colors)], s=20)



ax.set_xlabel("k")
ax.set_ylabel("$\lambda(k)$")
ax.set_xlim(0.7, 1.3)
ax.set_ylim(0.5, 2.0)
ax.axvspan(0.0, 0.99, color='gray', alpha=0.2, label="Single Mode Region")
ax.axvline(0.99, color='k', ls='--', lw=1, label="Cutoff $k_c = 0.99$")
# Generate legend handles for modes
handles, labels = ax.get_legend_handles_labels()
for j in range(2):
    handles.append(mpl.lines.Line2D([], [], marker="o", color=colors[j % len(colors)], linestyle="None", markersize=5, label=f"Mode {j+1}"))
ax.legend(handles=handles, title="Modes", loc="upper left", title_fontproperties={"weight" : "medium"})


# plt.suptitle("Dispersion Relation $\lambda(k)$ for Fiber Modes", y=0.95)
plt.suptitle("Zoom-In for Single Mode Region", y=0.95)
par_str = f"Evaluated with: $N={N}$, $x_{{max}}={x_max}$, $N_{{scan}}={n_scan}$, $\log_{{10}}(g)={int(np.log10(granularity))}$"
plt.title(par_str, fontsize=12)
plt.tight_layout()
plt.savefig(f"./Images/fiber_dispersion_k{k_start}_to_{k_end}_N{N}_x{x_max}_g{int(np.log10(granularity))}_zoom.png", dpi=450)
plt.show()
