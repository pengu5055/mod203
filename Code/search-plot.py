"""
Demonstrate residual root finding for the shooting method.
"""
import numpy as np
from src import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr

mpl.style.use("./vrm.mplstyle")
mpl.use("tkagg")

E = np.genfromtxt("./Data/Legacy_E_arr.csv", delimiter=",")
data = np.genfromtxt("./Data/Legacy_residual_arr.csv", delimiter=",")
eigenenergies = [9.8696044, 39.4784176, 88.82643961, 157.91367042, 246.74011003, 355.30575844, 483.61061565]

fig, ax = plt.subplots(figsize=(8, 6))
colors = cmr.take_cmap_colors("cmr.tropical", len(eigenenergies), cmap_range=(0.0, 0.8))

ax.plot(E, data, color='k', lw=2, zorder=3)

for i, energy in enumerate(eigenenergies):
    ax.scatter(energy, 0, color=colors[i], zorder=5, label=f"$E_{i}={energy:.2f}$", s=50, edgecolor='k', lw=0.25)
ax.axhline(0, color="black", lw=1, alpha=0.75, zorder=1)
plt.suptitle("Residuals of the Shooting Method\nfor an Infinite Potential Well")
ax.set_xlabel("Energy")
ax.set_ylabel("Residual")

ax.legend(loc="upper right", ncol=1, frameon=True, 
           prop={"size": 10, "weight": "medium"}, columnspacing=0.1, handletextpad=0.5,
           title="Eigenenergies", title_fontproperties={"size": 12, "weight": "bold"})
plt.tight_layout()
plt.savefig(f"./Images/shooting_residuals.png", dpi=450)
plt.show()

