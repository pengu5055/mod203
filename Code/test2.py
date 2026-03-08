"""
Test WIP Code part 2.
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

ts = time()

r_max = 500
granularity = 100000
r_range = np.linspace(1e-4, r_max, granularity)
