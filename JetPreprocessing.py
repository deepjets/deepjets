#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from skimage.transform import rotate
from shutil import copyfile
from subprocess import call

def ReadJetFiles(filename):
    """Return jet, constituent details as structured numpy arrays from file.
    
       Pythia jet format: pT, eta, phi, size.
       Pythia cst format: E,  ET,  eta, phi."""
    
    with open('{0}_jets.csv'.format(filename), 'r') as f:
        all_jets = np.genfromtxt(f, delimiter=',', usecols=(0,1,2), names=('pT','eta','phi'))
    
    with open('{0}_csts.csv'.format(filename), 'r') as f:
        jet_csts = np.genfromtxt(f, delimiter=',', usecols=(1,2,3), names=('ET','eta','phi'))
    
    return all_jets, jet_csts

def JetMass(jet_csts):
    """Returns jet mass calculated from constituent 4-vectors."""
    
    E_tot  = np.sum( jet_csts['ET']*np.cosh(jet_csts['eta']) )
    px_tot = np.sum( jet_csts['ET']*np.cos(jet_csts['phi']) )
    py_tot = np.sum( jet_csts['ET']*np.sin(jet_csts['phi']) )
    pz_tot = np.sum( jet_csts['ET']*np.sinh(jet_csts['eta']) )
    
    return np.sqrt( E_tot**2-px_tot**2-py_tot**2-pz_tot**2 )

def TranslateCsts(jet_csts, all_jets):
    """Return translated constituents, leading subjet at (eta, phi) = (0, 0)."""
    
    jet_csts['eta'] -= all_jets['eta'][1]
    jet_csts['phi'] -= all_jets['phi'][1]
    # Ensure phi in [-pi, pi]
    jet_csts['phi']  = np.mod(jet_csts['phi'] + np.pi, 2*np.pi) - np.pi
    
    return jet_csts

def TranslateJets(all_jets):
    """Return translated subjets, leading subjet centred at (eta, phi) = (0, 0)."""
    
    all_jets['eta'] -= all_jets['eta'][1]
    all_jets['phi'] -= all_jets['phi'][1]
    # Ensure phi in [-pi, pi]
    all_jets['phi']  = np.mod(all_jets['phi'] + np.pi, 2*np.pi) - np.pi
    
    return all_jets

def InitPixels(etaphi_range=(-1.25, 1.25, -1.25, 1.25), etaphi_delta=(0.1, 0.1)):
    """Returns blank pixel array and eta-phi bin edges."""
    
    eta_edges = np.arange(etaphi_range[0], etaphi_range[1]+0.1*etaphi_delta[0], etaphi_delta[0])
    phi_edges = np.arange(etaphi_range[2], etaphi_range[3]+0.1*etaphi_delta[1], etaphi_delta[1])
    pixels    = np.zeros(( len(eta_edges)-1, len(phi_edges)-1 ))
    
    return (eta_edges, phi_edges, pixels)
    
def Pixelise(jet_csts, eta_edges, phi_edges):
    """Return eta-phi histogram of transverse energy deposits."""
    
    pixels, eta_edges, phi_edges = np.histogram2d(jet_csts['phi'], jet_csts['eta'],
                                                  bins=(eta_edges, phi_edges),
                                                  weights=jet_csts['ET'])
    
    return pixels
    
def RotateJet(pixels, all_jets=[0.0]):
    """Return rotated and repixelised image array.
    
       Rotation puts subleading subjet or first principle component at -pi/2.
       Repixelisation interpolates with cubic spline."""
    
    if len(all_jets) > 2:
        # Use subleading subject information to rotate
        theta = np.arctan2(all_jets['phi'][2], all_jets['eta'][2])
        theta = (90.0+theta*180.0/np.pi)
        
        return rotate(pixels, theta, order=3)

    # Use principle component of image intensity to rotate
    width, height  = pixels.shape
    pix_coords     = np.array([ [i, j] for j in range(-height+1, height, 2)
                                       for i in range(-width+1, width, 2) ])
    covX           = np.cov( pix_coords, rowvar=0, aweights=np.reshape(pixels, (width*height)), bias=1 )
    e_vals, e_vecs = np.linalg.eigh(covX)
    pc             = e_vecs[:,-1]
    theta          = np.arctan2(pc[1], pc[0])
    theta          = (90+theta*180.0/np.pi)
    t_pixels       = rotate(pixels, theta, order=3)
    # Check orientation of principle component
    pix_bot = np.sum( t_pixels[:-(-height//2)].flatten() )
    pix_top = np.sum( t_pixels[(height//2):].flatten() )
    
    if pix_top > pix_bot:
        t_pixels = rotate(t_pixels, 180.0, order=3)
        theta    = theta+180.0
    
    return t_pixels

def ReflectJet(pixels, all_jets=[0.0]):
    """Return reflected image array.
    
       Reflection puts subsubleading subjet or highest intensity on right side."""
    
    width, height = pixels.shape
    
    if len(all_jets) > 3:
        # Use subleading subject information to rotate
        theta  = np.arctan2(all_jets['phi'][2], all_jets['eta'][2])
        theta  = (np.pi/2)+theta
        parity = np.sign( np.cos(-theta)*all_jets['eta'][3] - np.sin(-theta)*all_jets['phi'][3] )
        
    else:
        pix_l  = np.sum( pixels[:,:-(-width//2)].flatten() )
        pix_r  = np.sum( pixels[:,(width//2):].flatten() )
        parity = np.sign(pix_r - pix_l)
    
    if parity >= 0:
        return pixels
    
    t_pixels = np.array(pixels)
    
    for i in range(width):
        t_pixels[:,i] = pixels[:,-i]
    
    return t_pixels

def NormaliseJet(pixels):
    """Return normalised image array: sum(I**2) == 1."""
    
    sum_I = np.sum(pixels**2)
    
    return pixels / sum_I

def ShowJetImage(pixels, etaphi_range=(-1.25, 1.25, -1.25, 1.25), vmin=1e-9, vmax=1e3):
    """Displays jet image."""    
    
    fig = plt.figure(figsize=(5, 5))
    ax  = fig.add_subplot(111)
    p   = ax.imshow(pixels, extent=etaphi_range, origin='low',
                    interpolation='none', norm=LogNorm(vmin=vmin, vmax=vmax))
    fig.colorbar(p, ax=ax)
    fig.tight_layout()
    plt.show()
    fig.savefig('AvJetImage.pdf', format='pdf')
    plt.close()

etaphi_range = (-1.25, 1.25, -1.25, 1.25)
etaphi_delta = (0.1, 0.1)
njets        = 1000
generate     = False
normalise    = False
test         = False

jet_masses                   = np.zeros(njets)
eta_edges, phi_edges, pixels = InitPixels(etaphi_range, etaphi_delta)
t_pixels                     = np.array(pixels)

for i in range(njets):
    
    if test:
        break
    
    if generate:
        call('./WprimeJetGen > out.txt'.split())
        copyfile('{0}_jets.csv'.format('test'), 'JetData/{0}_jets.csv'.format(i))
        copyfile('{0}_csts.csv'.format('test'), 'JetData/{0}_csts.csv'.format(i))
        all_jets, jet_csts = ReadJetFiles('test')
        
    else:
        all_jets, jet_csts = ReadJetFiles('JetData/{0}'.format(i))
    
    jet_masses[i] = JetMass(jet_csts)
    jet_csts      = TranslateCsts(jet_csts, all_jets)
    all_jets      = TranslateJets(all_jets)
    pixels_i      = Pixelise(jet_csts, eta_edges, phi_edges)
    pixels       += pixels_i
    t_pixels_i    = RotateJet(pixels_i, all_jets)
    t_pixels_i    = ReflectJet(t_pixels_i, all_jets)
    
    if normalise:
        t_pixels_i = NormaliseJet(t_pixels_i)
    
    t_pixels  += t_pixels_i

if test:
    eta_edges, phi_edges, pixels = InitPixels(etaphi_range, etaphi_delta)
    all_jets, jet_csts           = ReadJetFiles('TestJetData/3')
    
    pixels   = Pixelise(jet_csts, eta_edges, phi_edges)
    print "\n\nOriginal"
    ShowJetImage(pixels, vmin=0.09, vmax=11.0)
    
    jet_csts = TranslateCsts(jet_csts, all_jets)
    all_jets = TranslateJets(all_jets)
    pixels   = Pixelise(jet_csts, eta_edges, phi_edges)
    print "Translated"
    ShowJetImage(pixels, vmin=0.09, vmax=11.0)
    
    pixels   = RotateJet(pixels, all_jets)
    print "Rotated"
    ShowJetImage(pixels, vmin=0.09, vmax=11.0)
    
    pixels   = ReflectJet(pixels, all_jets)
    print "Reflected"
    ShowJetImage(pixels, vmin=0.09, vmax=11.0)

else:
    ShowJetImage(pixels / njets)
    ShowJetImage(t_pixels / njets)
    
    fig = plt.figure(figsize=(5, 5))
    ax  = fig.add_subplot(111)
    p   = ax.hist(jet_masses, bins=25, range=(40, 120))
    fig.tight_layout()
    plt.show()
    plt.close()