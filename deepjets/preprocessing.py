import numpy as np
from skimage.transform import rotate


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
    if slsubjet_centre is None:
        theta = 0.0
    elif slsubjet_centre[0] == 0.0:
        theta = np.arctan2(slsubjet_centre[1], slsubjet_centre[0])
    else:
        theta = np.arctan2(slsubjet_centre[1], slsubjet_centre[0])

    return rotate(pixels, -theta*180.0/np.pi, order=3)
