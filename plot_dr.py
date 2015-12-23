
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import h5py

hists = []
for file in glob('w_*_images.h5'):
    with h5py.File(file, 'r') as infile:
        dset_images = infile['images']
        hist, _, _ = np.histogram2d(
                dset_images['pT'], dset_images['dR_subjets'],
                bins=(np.linspace(150, 500, 31), np.linspace(0.3, 1.2, 31)))
        hists.append(hist)

hist = np.sum(hists, axis=0)
hist /= hist.sum(axis=1)[:, np.newaxis]

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111)
ax.imshow(hist.T, cmap='jet', extent=(150, 500, 0.3, 1.2),
          interpolation='nearest',
          origin='low',
          aspect='auto')
ax.set_ylabel(r'$\Delta R$', fontsize=18)
ax.set_xlabel(r'$p_{T}$', fontsize=18)
fig.tight_layout()
fig.savefig('dR.png')
