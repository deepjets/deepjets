#!/usr/bin/env python


def plot(input):
    import os
    import numpy as np
    from deepjets import generate
    from deepjets.preprocessing import preprocess
    from deepjets.utils import plot_jet_image, jet_mass
    from matplotlib import pyplot as plt
    import h5py
    
    print("plotting {0} ...".format(input))

    h5file = h5py.File(input, 'r')
    dset_images = h5file['images']
    h5file_events = h5py.File(input.replace('_images', ''), 'r')
    dset_jet = h5file_events['jet']
    dset_trimmed_pt = h5file_events['trimmed_pT']
    dset_trimmed_mass = h5file_events['trimmed_mass']

    output_prefix = os.path.splitext(input)[0]

    # plot jet images
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    avg_image = dset_images[:].sum(axis=0) / len(dset_images)
    plot_jet_image(ax, avg_image, vmax=1e-2)
    fig.tight_layout()
    fig.savefig(output_prefix + '.png')

    fig = plt.figure(figsize=(5, 5))
    ax  = fig.add_subplot(111)
    ax.hist(dset_jet['mass'], bins=np.linspace(0, 120, 20),
            histtype='stepfilled', facecolor='none', edgecolor='blue')
    fig.tight_layout()
    plt.savefig(output_prefix + '_jet_mass.png')
    
    fig = plt.figure(figsize=(5, 5))
    ax  = fig.add_subplot(111)
    ax.hist(dset_trimmed_mass[:], bins=np.linspace(0, 120, 20),
            histtype='stepfilled', facecolor='none', edgecolor='blue')
    fig.tight_layout()
    plt.savefig(output_prefix + '_jet_trimmed_mass.png')

    fig = plt.figure(figsize=(5, 5))
    ax  = fig.add_subplot(111)
    ax.hist(dset_jet['pT'], bins=np.linspace(0, 600, 100),
            histtype='stepfilled', facecolor='none', edgecolor='blue')
    fig.tight_layout()
    plt.savefig(output_prefix + '_jet_pt.png')
    
    fig = plt.figure(figsize=(5, 5))
    ax  = fig.add_subplot(111)
    ax.hist(dset_trimmed_pt[:], bins=np.linspace(0, 600, 100),
            histtype='stepfilled', facecolor='none', edgecolor='blue')
    fig.tight_layout()
    plt.savefig(output_prefix + '_jet_trimmed_pt.png')

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-n', type=int, default=-1)
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()

    from deepjets.parallel import map_pool, FuncWorker

    map_pool(
        FuncWorker, [(plot, filename) for filename in args.files],
        n_jobs=args.n)
