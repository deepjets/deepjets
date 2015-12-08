import numpy as np
from deepjets import generate
from deepjets.preprocessing import preprocess
from deepjets.utils import plot_jet_image

eta_edges = np.linspace(-1.2, 1.2, 26)
phi_edges = np.linspace(-1.2, 1.2, 26)
nevents = 1000

# W'
pixels = np.zeros((len(eta_edges) - 1, len(phi_edges) - 1))
for leading_jet, subjets, constit in generate('wprime.config', nevents,
                                             w_pt_min=250, w_pt_max=300):
    pixels += preprocess(leading_jet, subjets, constit, eta_edges, phi_edges)
plot_jet_image(pixels / nevents, eta_edges, phi_edges, filename='wprime.png')

# QCD dijets
pixels = np.zeros((len(eta_edges) - 1, len(phi_edges) - 1))
for leading_jet, subjets, constit in generate('qcd.config', nevents):
    pixels += preprocess(leading_jet, subjets, constit, eta_edges, phi_edges)
plot_jet_image(pixels / nevents, eta_edges, phi_edges, filename='qcd.png')
