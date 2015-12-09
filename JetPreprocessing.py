#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from skimage.transform import rotate
from shutil import copyfile
from subprocess import call

def ReadJetFiles(filename):
    """Return jet, constituent details as numpy arrays from file.
    
       Jet format: pT, eta, phi, size.
       Cst format: E,  Et,  eta, phi."""
    
    with open('{0}_jets.csv'.format(filename), 'r') as f:
        all_jets = np.genfromtxt(f, delimiter=',')
    
    with open('{0}_csts.csv'.format(filename), 'r') as f:
        all_csts = np.genfromtxt(f, delimiter=',')
    
    return all_jets, all_csts

def TranslateCsts(jet_csts, centre):
    """Return translated constituents, centred at (eta, phi) = (0, 0)."""
    
    x_centre  = np.array([0, 0] + list(centre))
    jet_csts -= x_centre
    return jet_csts

def Pixelise(jet_csts, eta_edges, phi_edges):
    """Return eta-phi histogram of transverse energy deposits."""
    
    pixels, eta_edges, phi_edges = np.histogram2d(jet_csts[:,2], jet_csts[:,3],
                                                  bins=(eta_edges, phi_edges),
                                                  weights=jet_csts[:,1])
    return pixels
    
def RotateJet(pixels, slsubjet_centre=[0.0]):
    """Return rotated and repixelised image array.
    
       Rotation puts subleading subjet or first principle component at -pi/2.
       Repixelisation interpolates with cubic spline."""
    
    if len(slsubjet_centre) == 2:
        # Use subleading subject information
        # Difference of 90 degrees due to pixel ordering
        theta = np.arctan2(slsubjet_centre[1], slsubjet_centre[0])
        return rotate(pixels, -(theta*180.0/np.pi), order=3)

    # Find principle component of image intensity
    # Difference of 90 degrees due to pixel ordering
    width, height  = pixels.shape
    pix_coords     = np.array([ [i, j] for j in range(height-1, -height, -2)
                                       for i in range(-width+1, width, 2) ])
    covX           = np.cov( pix_coords, rowvar=0, aweights=np.reshape(pixels, (width*height)), bias=1 )
    e_vals, e_vecs = np.linalg.eigh(covX)
    pc             = e_vecs[:,-1]
    theta          = np.arctan2(pc[1], pc[0])
    t_pixels       = rotate(pixels, -(90+theta*180.0/np.pi), order=3)
    
    # Check orientation of principle component
    pix_top = sum( t_pixels[:-(-height//2)].flatten() )
    pix_bot = sum( t_pixels[(height//2):].flatten() )
    if pix_top > pix_bot:
        t_pixels = rotate(t_pixels, 180.0, order=3)
    
    return t_pixels

def ReflectJet(pixels):
    """Return reflected image array.
    
       Reflection ensures right side of image has highest intensity."""
    
    width, height  = pixels.shape
    pix_l = sum( pixels[:,:-(-width//2)].flatten() )
    pix_r = sum( pixels[:,(width//2):].flatten() )
    if pix_l < pix_r:
        return pixels
    
    t_pixels = np.array(pixels)
    for i in range(width):
        t_pixels[:,i] = pixels[:,-i]
    return t_pixels

def NormaliseJet(pixels):
    """Return normalised image array: sum(I**2) == 1."""
    
    sum_I = np.sum(pixels**2)
    return pixels / sum_I

def JetMass(jet_csts):
    """Returns jet mass calculated from constituent 4-vectors."""
    
    E_tot  = np.sum( jet_csts[:,1]*np.cosh(jet_csts[:,2]) )
    px_tot = np.sum( jet_csts[:,1]*np.cos(jet_csts[:,3]) )
    py_tot = np.sum( jet_csts[:,1]*np.sin(jet_csts[:,3]) )
    pz_tot = np.sum( jet_csts[:,1]*np.sinh(jet_csts[:,2]) )
    
    return np.sqrt( E_tot**2-px_tot**2-py_tot**2-pz_tot**2 )

def ShowJetImage(pixels, etaphi_range=(-1.25, 1.25, -1.25, 1.25), vmin=1e-9, vmax=1e3):
    fig  = plt.figure(figsize=(5, 5))
    ax   = fig.add_subplot(111)
    p    = ax.imshow(pixels, extent=etaphi_range, interpolation='none', norm=LogNorm(vmin=vmin, vmax=vmax))
    fig.colorbar(p, ax=ax)
    fig.tight_layout()
    plt.show()
    fig.savefig('AvJetImage.pdf', format='pdf')
    plt.close()

  
etaphi_range = (-1.25, 1.25, -1.25, 1.25)
etaphi_delta = (0.1, 0.1)
njets        = 1000
jet_masses   = np.zeros(njets)
eta_edges    = np.arange(etaphi_range[0], 1.01*etaphi_range[1], etaphi_delta[0])
phi_edges    = np.arange(etaphi_range[2], 1.01*etaphi_range[3], etaphi_delta[1])
pixels       = np.zeros(( len(eta_edges)-1, len(phi_edges)-1 ))
t_pixels     = np.zeros(( len(eta_edges)-1, len(phi_edges)-1 ))
generate     = False

for i in range(njets):
    if generate:
        call('./WprimeJetGen > out.txt'.split())
        copyfile('{0}_jets.csv'.format('test'), 'JetData/{0}_jets.csv'.format(i))
        copyfile('{0}_csts.csv'.format('test'), 'JetData/{0}_csts.csv'.format(i))
        all_jets, jet_csts = ReadJetFiles('test')
    else:
        all_jets, jet_csts = ReadJetFiles('JetData/{0}'.format(i))
    jet_masses[i]      = JetMass(jet_csts)
    lsubjet_centre     = all_jets[1][1:3] 
    jet_csts           = TranslateCsts(jet_csts, lsubjet_centre)
    jet_pixels         = Pixelise(jet_csts, eta_edges, phi_edges)
    if len(all_jets) > 2:
        slsubjet_centre = all_jets[2][1:3] - lsubjet_centre
    else:
        slsubjet_centre = [0.0]
    pixels   += jet_pixels
    t_pix     = RotateJet(jet_pixels, slsubjet_centre)
    t_pix     = ReflectJet(t_pix)
    t_pix     = NormaliseJet(t_pix)
    t_pixels += t_pix

ShowJetImage(pixels / njets)
ShowJetImage( t_pixels / njets, vmin=1e-9, vmax=np.amax(t_pixels / njets) )

fig  = plt.figure(figsize=(5, 5))
ax   = fig.add_subplot(111)
p    = ax.hist(jet_masses, bins=20)
fig.tight_layout()
plt.show()
plt.close()