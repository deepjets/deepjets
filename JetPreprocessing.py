#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LogNorm
from skimage.transform import rotate
from shutil import copyfile
from subprocess import call

def ReadJetFiles(filename):
    """Return jet, constituent details as numpy arrays from file
    
       Jet format: pT, eta, phi, size
       Cst format: E,  Et,  eta, phi"""    
    with open('{0}_jets.csv'.format(filename), 'r') as f:
        all_jets = np.genfromtxt(f, delimiter=',')
    
    with open('{0}_csts.csv'.format(filename), 'r') as f:
        all_csts = np.genfromtxt(f, delimiter=',')
    
    return all_jets, all_csts

def TranslateCsts(jet_csts, centre):
    """Return translated constituents, centred at (eta, phi) = (0, 0)"""
    x_centre  = np.array([0, 0] + list(centre))
    jet_csts -= x_centre
    return jet_csts

def Pixelise(jet_csts, eta_edges, phi_edges):
    """Return eta-phi histogram of transverse energy deposits"""
    pixels, eta_edges, phi_edges = np.histogram2d(jet_csts[:,2], jet_csts[:,3],
                                                  bins=(eta_edges, phi_edges),
                                                  weights=jet_csts[:,1])
    return pixels
    
def RotateJet(pixels, slsubjet_centre=None):
    """Return rotated and repixelised eta-phi histogram
    
       Rotation puts subleading subjet or first principle component at -pi/2
       Repixelisation interpolates with cubic spline"""
    if slsubjet_centre == None:
        theta = 0.0
    elif slsubjet_centre[0] == 0.0:
        theta = np.arctan2(slsubjet_centre[1], slsubjet_centre[0])
    else:
        theta = np.arctan2(slsubjet_centre[1], slsubjet_centre[0])
    
    return rotate(pixels, -theta*180.0/np.pi, order=3)

def ShowJetImage(pixels, etaphi_range=(-1.25, 1.25, -1.25, 1.25)):
    fig  = plt.figure(figsize=(5, 5))
    ax   = fig.add_subplot(111)
    p    = ax.imshow(pixels, extent=etaphi_range, interpolation='none', norm=LogNorm(vmin=1e-9, vmax=1e3))
    fig.colorbar(p, ax=ax)
    fig.tight_layout()
    plt.show()
    fig.savefig('AvJetImage.pdf', format='pdf')
    plt.close()

"""    
test_pix    = np.array([[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]])
test_centre = np.array([-1,0])
print RotateJet(test_pix, test_centre)  
"""  

os.environ['DYLD_LIBRARY_PATH'] = '/Users/barney800/Tools/fastjet/lib'

etaphi_range = (-1.2, 1.2, -1.2, 1.2)
etaphi_delta = (0.1, 0.1)
eta_edges    = np.arange(etaphi_range[0], 1.01*etaphi_range[1], etaphi_delta[0])
phi_edges    = np.arange(etaphi_range[2], 1.01*etaphi_range[3], etaphi_delta[1])
pixels       = np.zeros(( len(eta_edges) - 1, len(phi_edges) - 1 ))
r_pixels     = np.zeros(( len(eta_edges) - 1, len(phi_edges) - 1 ))
njets        = 1000
generate     = False
for i in range(njets):
    if generate:
        call('./WprimeJetGen > out.txt'.split())
        copyfile('{0}_jets.csv'.format('test'), 'JetData/{0}_jets.csv'.format(i))
        copyfile('{0}_csts.csv'.format('test'), 'JetData/{0}_csts.csv'.format(i))
        all_jets, jet_csts = ReadJetFiles('test')
    else:
        all_jets, jet_csts = ReadJetFiles('JetData/{0}'.format(i))
    lsubjet_centre     = all_jets[1][1:3] 
    jet_csts           = TranslateCsts(jet_csts, lsubjet_centre)
    jet_pixels         = Pixelise(jet_csts, eta_edges, phi_edges)
    if len(all_jets) > 2:
        slsubjet_centre = all_jets[2][1:3] - lsubjet_centre
    else:
        slsubjet_centre = None
    pixels   += jet_pixels
    r_pixels += RotateJet(jet_pixels, slsubjet_centre)
ShowJetImage(pixels / njets)
ShowJetImage(r_pixels / njets)