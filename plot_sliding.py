import matplotlib.pyplot as plt
from matplotlib import animation
from glob import glob
import h5py
from deepjets.utils import plot_jet_image
import numpy as np

eta_edges = np.linspace(-2, 2, 40)
phi_edges = np.linspace(-2, 2, 40)

fig = plt.figure(figsize=(6, 5))


def get_images(filenames, min_pt, max_pt):
    images = []
    for filename in filenames:
        print "grabbing images from {0}".format(filename)
        with h5py.File(filename, 'r') as h5file:
            dset_images = h5file['images']
            images.append(dset_images['image'][(dset_images['pT'] > min_pt) &
                                               (dset_images['pT'] <= max_pt)])
        print "got {0} images".format(images[-1].shape[0])
    return np.concatenate(images)


def animate(iframe):
    print "plotting frame {0}".format(iframe)
    plt.clf()
    ax = fig.add_subplot(111)
    images = get_images(glob('w_*_images.h5'),
                        150 + 20 * iframe, 160 + 20 * iframe)
    print "total of {0} images".format(len(images))
    avg_image = images.sum(axis=0) / len(images)
    plot_jet_image(ax, avg_image, eta_edges, phi_edges)


anim = animation.FuncAnimation(fig, animate, frames=30)
anim.save('images.gif', writer='imagemagick', fps=4)
