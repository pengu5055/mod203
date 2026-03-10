"""
Test FPW code.
"""
import numpy as np
from src import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
import os
from time import time
from scipy.optimize import brentq

mpl.style.use("./vrm.mplstyle")
mpl.use("tkagg")

def even_eq(E, V0, a):
    k = np.sqrt(E)
    kappa = np.sqrt((V0-E))
    return k*np.tan(k*a) - kappa

def odd_eq(E, V0, a):
    k = np.sqrt(E)
    kappa = np.sqrt((V0-E))
    return -k/np.tan(k*a) - kappa

V0, a = 100.0, 1.0
E_test = np.linspace(0.01, V0 - 0.01, 100000)

roots = []
for eq in [even_eq, odd_eq]:
    f = np.array([eq(E, V0, a) for E in E_test])
    sign_changes = np.where(np.diff(np.sign(f)))[0]
    for i in sign_changes:
        if np.abs(f[i+1] - f[i]) < 50:
            root = brentq(eq, E_test[i], E_test[i+1], args=(V0, a))
            roots.append(root)

roots = np.sort(roots)
print(roots)
