import numpy as np
from deepjets import generate
from deepjets.preprocessing import TranslateCsts, Pixelise, RotateJet, ShowJetImage

etaphi_range = (-1.2, 1.2, -1.2, 1.2)
etaphi_delta = (0.1, 0.1)
eta_edges    = np.arange(etaphi_range[0], 1.01*etaphi_range[1], etaphi_delta[0])
phi_edges    = np.arange(etaphi_range[2], 1.01*etaphi_range[3], etaphi_delta[1])
pixels       = np.zeros(( len(eta_edges) - 1, len(phi_edges) - 1 ))
r_pixels     = np.zeros(( len(eta_edges) - 1, len(phi_edges) - 1 ))
nevents = 1000

for leading_jet, subjets, constit in generate('wprime.config', nevents,
                                             w_pt_min=250, w_pt_max=300):
    lsubjet_centre = subjets[0][1:3]
    jet_csts = TranslateCsts(constit, lsubjet_centre)
    jet_pixels = Pixelise(constit, eta_edges, phi_edges)
    if len(subjets) > 1:
        slsubjet_centre = subjets[1][1:3] - lsubjet_centre
    else:
        slsubjet_centre = None
    pixels   += jet_pixels
    r_pixels += RotateJet(jet_pixels, slsubjet_centre)


#for leading_jet, subjets, constit in generate('qcd.config', 10):
#    print leading_jet, subjets.shape[0], constit.shape[0]

ShowJetImage(r_pixels / nevents)
#ShowJetImage(r_pixels / njets)
