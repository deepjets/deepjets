import numpy as np
from deepjets import generate
from deepjets.preprocessing import preprocess
from deepjets.utils import plot_jet_image, jet_mass
from matplotlib import pyplot as plt
import h5py

eta_edges = np.linspace(-1.2, 1.2, 26)
phi_edges = np.linspace(-1.2, 1.2, 26)
nevents = 1000

# W
jet_mass_wprime = []
pixels = np.zeros((len(eta_edges) - 1, len(phi_edges) - 1))

h5file = h5py.File('wprime.h5', 'r')
dset_jets = h5file['jets']
dset_constit = h5file['constituents']
dset_trimmed_constit = h5file['trimmed_constituents']

for event in xrange(len(dset_jets)):
    jets = dset_jets[event]
    constit = dset_constit[event]
    trimmed_constit = dset_trimmed_constit[event]
    jet_mass_wprime.append(jet_mass(trimmed_constit))
    pixels += preprocess(jets, trimmed_constit, eta_edges, phi_edges)
plot_jet_image(pixels / nevents, eta_edges, phi_edges, filename='wprime.png')

# QCD dijets
jet_mass_qcd = []
pixels = np.zeros((len(eta_edges) - 1, len(phi_edges) - 1))

h5file = h5py.File('qcd.h5', 'r')
dset_jets = h5file['jets']
dset_constit = h5file['constituents']
dset_trimmed_constit = h5file['trimmed_constituents']

for event in xrange(len(dset_jets)):
    jets = dset_jets[event]
    constit = dset_constit[event]
    trimmed_constit = dset_trimmed_constit[event]
    jet_mass_qcd.append(jet_mass(trimmed_constit))
    pixels += preprocess(jets, trimmed_constit, eta_edges, phi_edges)
plot_jet_image(pixels / nevents, eta_edges, phi_edges, filename='qcd.png')

# plot
fig = plt.figure(figsize=(5, 5))
ax  = fig.add_subplot(111)
ax.hist(jet_mass_qcd, bins=np.linspace(0, 120, 20), label='QCD',
        histtype='stepfilled', facecolor='none', edgecolor='red')
ax.hist(jet_mass_wprime, bins=np.linspace(0, 120, 20), label='W',
        histtype='stepfilled', facecolor='none', edgecolor='blue')
ax.legend()
fig.tight_layout()
plt.savefig('jet_mass.png')
