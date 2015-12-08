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

def RotateJet(pixels, slsubjet_centre=[0.0]):
    """Return rotated and repixelised image array

       Rotation puts subleading subjet or first principle component at -pi/2
       Repixelisation interpolates with cubic spline"""
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
    r_pixels       = rotate(pixels, -(90+theta*180.0/np.pi), order=3)

    # Check orientation of principle component
    pix_top = sum( r_pixels[:-(-height//2)].flatten() )
    pix_bot = sum( r_pixels[(height//2):].flatten() )
    if pix_top > pix_bot:
        r_pixels = rotate(r_pixels, 180.0, order=3)

    return r_pixels

def ReflectJet(pixels):
    """Return reflected image array

       Reflection ensures right side of image has highest intensity"""
    width, height  = pixels.shape
    pix_l = sum( pixels[:,:-(-width//2)].flatten() )
    pix_r = sum( pixels[:,(width//2):].flatten() )
    if pix_l < pix_r:
        return pixels

    r_pixels = np.array(pixels)
    for i in range(width):
        r_pixels[:,i] = pixels[:,-i]
    return r_pixels


def preprocess(jets, subjets, constit, eta_edges, phi_edges):
    lsubjet_centre     = subjets[0][1:3]
    constit            = TranslateCsts(constit, lsubjet_centre)
    jet_pixels         = Pixelise(constit, eta_edges, phi_edges)
    if len(subjets) > 1:
        slsubjet_centre = subjets[1][1:3] - lsubjet_centre
    else:
        slsubjet_centre = [0.0]
    return ReflectJet(RotateJet(jet_pixels, slsubjet_centre))
