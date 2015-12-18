#!/usr/bin/env python

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('images_file')
args = parser.parse_args()

import os
import numpy as np
from deepjets import generate
from deepjets.preprocessing import preprocess
from deepjets.utils import plot_jet_image, jet_mass
from matplotlib import pyplot as plt
import h5py

eta_edges = np.linspace(-1.2, 1.2, 26)
phi_edges = np.linspace(-1.2, 1.2, 26)
pixels = np.zeros((len(eta_edges) - 1, len(phi_edges) - 1))

h5file = h5py.File(args.images_file, 'r')
dset_images = h5file['images']
output_prefix = os.path.splitext(args.images_file)[0]
avg_image = dset_images['image'].sum(axis=0) / len(dset_images)
plot_jet_image(avg_image, eta_edges, phi_edges, filename=output_prefix + '.png')

# plot
fig = plt.figure(figsize=(5, 5))
ax  = fig.add_subplot(111)
ax.hist(dset_images['mass'], bins=np.linspace(0, 120, 20),
        histtype='stepfilled', facecolor='none', edgecolor='blue')
fig.tight_layout()
plt.savefig(output_prefix + '_jet_mass.png')

fig = plt.figure(figsize=(5, 5))
ax  = fig.add_subplot(111)
ax.hist(dset_images['pT'], bins=np.linspace(0, 120, 20),
        histtype='stepfilled', facecolor='none', edgecolor='blue')
fig.tight_layout()
plt.savefig(output_prefix + '_jet_pt.png')
