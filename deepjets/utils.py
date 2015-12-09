import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_jet_image(pixels, eta_edges, phi_edges, filename='jet_image.png'):
    eta_min, eta_max = eta_edges.min(), eta_edges.max()
    phi_min, phi_max = phi_edges.min(), phi_edges.max()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    p = ax.imshow(pixels, extent=(eta_min, eta_max, phi_min, phi_max),
                  interpolation='none',
                  norm=LogNorm(vmin=1e-9, vmax=1e3))
    fig.colorbar(p, ax=ax)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close()


def jet_mass(jet_csts):
    """Returns jet mass calculated from constituent 4-vectors."""
    E_tot  = np.sum( jet_csts[:,1]*np.cosh(jet_csts[:,2]) )
    px_tot = np.sum( jet_csts[:,1]*np.cos(jet_csts[:,3]) )
    py_tot = np.sum( jet_csts[:,1]*np.sin(jet_csts[:,3]) )
    pz_tot = np.sum( jet_csts[:,1]*np.sinh(jet_csts[:,2]) )
    return np.sqrt( E_tot**2-px_tot**2-py_tot**2-pz_tot**2 )
