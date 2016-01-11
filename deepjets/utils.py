import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_jet_image(ax, pixels, vmin=1e-9, vmax=1e-2):
    """Displays jet image."""
    p = ax.imshow(pixels.T, extent=(-1, 1, -1, 1),
                  origin='low',
                  interpolation='nearest',
                  norm=LogNorm(vmin=vmin, vmax=vmax),
                  cmap='jet')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(
        p, cax=cax,
        ticks=np.logspace(np.log10(vmin), np.log10(vmax), 1+np.log10(vmax)-np.log10(vmin)))
    cbar.set_label(r'Intensity', rotation=90, fontsize=18)
    ax.set_xlabel(r'$x_1$', fontsize=18)
    ax.set_ylabel(r'$x_2$', fontsize=18)


def jet_mass(jet_csts):
    """Returns jet mass calculated from constituent 4-vectors."""
    E_tot  = np.sum(jet_csts['ET'] * np.cosh(jet_csts['eta']))
    px_tot = np.sum(jet_csts['ET'] * np.cos(jet_csts['phi']))
    py_tot = np.sum(jet_csts['ET'] * np.sin(jet_csts['phi']))
    pz_tot = np.sum(jet_csts['ET'] * np.sinh(jet_csts['eta']))
    m2 = E_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2
    if m2 <= 0:
        return 0.
    return np.sqrt(E_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2)
