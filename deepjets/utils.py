import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def load_images(image_h5_file, n_images=-1, aux_vars=[], shuffle=False):
    """Loads images from h5 file.
    
    Optionally choose number of images to load and whether to shuffle on loading.
    TODO: test support for additional fields.
    """
    with h5py.File(image_h5_file, 'r') as h5file:
        images = h5file['images']['image']
        aux_data = {var : h5file['aux_vars'][var] for var in aux_vars}
    if shuffle:
        np.random.shuffle(images)
    if n_images < 0:
        n_images = len(images)
    elif n_images > len(images):
        print 'Cannot load {0} images from {1}, only {2} images present.'.format(
            n_images, image_h5_file, len(images))
        n_images = len(images)
    images = images[:n_images]
    return (images, aux_data)
    

def plot_jet_image(ax, image, vmin=1e-9, vmax=1e-2):
    """Displays jet image."""
    width, height = image.T.shape
    dw, dh = 1./width, 1./height
    p = ax.imshow(image.T, extent=(-(1+dw), 1+dw, -(1+dh), 1+dh),
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


def tot_mom(jet_csts):
    E_tot  = np.sum(jet_csts['ET'] * np.cosh(jet_csts['eta']))
    px_tot = np.sum(jet_csts['ET'] * np.cos(jet_csts['phi']))
    py_tot = np.sum(jet_csts['ET'] * np.sin(jet_csts['phi']))
    pz_tot = np.sum(jet_csts['ET'] * np.sinh(jet_csts['eta']))
    return E_tot, px_tot, py_tot, pz_tot


def mass(E, px, py, pz):
    m2 = E**2 - px**2 - py**2 - pz**2
    return np.sign(m2) * np.sqrt(abs(m2))


def jet_mass(jet_csts):
    """Returns jet mass calculated from constituent 4-vectors."""
    return mass(*tot_mom(jet_csts))


def pT(E, px, py, pz):
    return (px**2 + py**2)**0.5


dphi = lambda phi1, phi2 : abs(math.fmod((math.fmod(phi1, 2*math.pi) - math.fmod(phi2, 2*math.pi)) + 3*math.pi, 2*math.pi) - math.pi)


def dR(jet1, jet2):
    return ((jet1['eta'] - jet2['eta'])**2 + dphi(jet1['phi'], jet2['phi'])**2)**0.5
