import numpy as np
from deepjets import generate
from deepjets.preprocessing import preprocess
from deepjets.utils import plot_jet_image, jet_mass
from matplotlib import pyplot as plt

eta_edges = np.linspace(-1.2, 1.2, 26)
phi_edges = np.linspace(-1.2, 1.2, 26)
nevents = 200

# W'
jet_mass_wprime = []
pixels = np.zeros((len(eta_edges) - 1, len(phi_edges) - 1))
for event in generate('wprime.config', nevents, jet_size=1.2):
    leading_jet, constit, subjets, trimmed_constit = event
    jet_mass_wprime.append(jet_mass(trimmed_constit))
    pixels += preprocess(leading_jet, subjets, trimmed_constit, eta_edges, phi_edges)
plot_jet_image(pixels / nevents, eta_edges, phi_edges, filename='wprime.png')

# QCD dijets
jet_mass_qcd = []
pixels = np.zeros((len(eta_edges) - 1, len(phi_edges) - 1))
for event in generate('qcd.config', nevents, jet_size=1.2):
    leading_jet, constit, subjets, trimmed_constit = event
    jet_mass_qcd.append(jet_mass(trimmed_constit))
    pixels += preprocess(leading_jet, subjets, trimmed_constit, eta_edges, phi_edges)
plot_jet_image(pixels / nevents, eta_edges, phi_edges, filename='qcd.png')

fig = plt.figure(figsize=(5, 5))
ax  = fig.add_subplot(111)
ax.hist(jet_mass_qcd, bins=np.linspace(0, 120, 20), label='QCD',
        histtype='stepfilled', facecolor='none', edgecolor='red')
ax.hist(jet_mass_wprime, bins=np.linspace(0, 120, 20), label='W',
        histtype='stepfilled', facecolor='none', edgecolor='blue')
ax.legend()
fig.tight_layout()
plt.savefig('jet_mass.png')
