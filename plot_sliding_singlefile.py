#!/usr/bin/env python
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('filename')
parser.add_argument('output')
args = parser.parse_args()

import matplotlib.pyplot as plt
from matplotlib import animation
import h5py
from deepjets.utils import plot_jet_image
import numpy as np

hfile = h5py.File(args.filename, 'r')
images = hfile['images'][:]
weights = hfile['auxvars']['weights']
pt = hfile['auxvars']['pt_trimmed']

fig = plt.figure(figsize=(6, 5))


def animate(iframe):
    print "plotting frame {0}".format(iframe)
    plt.clf()
    ax = fig.add_subplot(111)
    min_pt = 200 + 10 * iframe
    max_pt = 220 + 10 * iframe
    window_cond = (pt > min_pt) & (pt <= max_pt)
    images_in_window = images[np.where(window_cond)]
    weights_in_window = weights[np.where(window_cond)]
    avg_image = np.average(images_in_window, axis=0, weights=weights_in_window)
    print "total of {0} images selected".format(len(images_in_window))
    plot_jet_image(ax, avg_image)


anim = animation.FuncAnimation(fig, animate, frames=25)
anim.save(args.output, writer='imagemagick', fps=4)
