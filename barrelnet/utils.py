import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


def cmapvals(vals, cmap="viridis", vmin=None, vmax=None):
    cmap = plt.get_cmap(cmap)
    if vmin is None:
        vmin = np.min(vals)
    if vmax is None:
        vmax = np.max(vals)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgbvals = np.array(scalarMap.to_rgba(vals))
    rgbvals = rgbvals[:, :3]
    return rgbvals
