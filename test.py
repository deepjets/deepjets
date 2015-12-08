import numpy as np
from deepjets import generate
from deepjets.preprocessing import preprocess
from deepjets.utils import plot_jet_image

etaphi_range = (-1.2, 1.2, -1.2, 1.2)
etaphi_delta = (0.1, 0.1)
eta_edges    = np.arange(etaphi_range[0], 1.01*etaphi_range[1], etaphi_delta[0])
phi_edges    = np.arange(etaphi_range[2], 1.01*etaphi_range[3], etaphi_delta[1])
nevents = 5000

# W'
pixels = np.zeros(( len(eta_edges) - 1, len(phi_edges) - 1 ))
for leading_jet, subjets, constit in generate('wprime.config', nevents,
                                             w_pt_min=250, w_pt_max=300):
    pixels += preprocess(leading_jet, subjets, constit, eta_edges, phi_edges)
plot_jet_image(pixels / nevents, filename='wprime.png')

# QCD dijets
pixels = np.zeros(( len(eta_edges) - 1, len(phi_edges) - 1 ))
for leading_jet, subjets, constit in generate('qcd.config', nevents):
    pixels += preprocess(leading_jet, subjets, constit, eta_edges, phi_edges)
plot_jet_image(pixels / nevents, filename='qcd.png')
