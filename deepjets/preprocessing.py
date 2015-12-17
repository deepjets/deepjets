import numpy as np
from skimage.transform import rotate


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


def Pixelise(jet_csts, eta_edges, phi_edges, cutoff=0.0):
    """Return eta-phi histogram of transverse energy deposits.

       Optionally set all instensities below cutoff to zero."""

    weights = [ ET if ET > cutoff else 0.0 for ET in jet_csts['ET'] ]
    pixels, eta_edges, phi_edges = np.histogram2d(jet_csts['phi'], jet_csts['eta'],
                                                  bins=(eta_edges, phi_edges),
                                                  weights=weights)

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
        # Use subsubleading subject information to reflect
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


def preprocess(jets, constit, eta_edges, phi_edges,
               cutoff=0.1, normalise=False):
    constit = TranslateCsts(constit, jets)
    jets = TranslateJets(jets)
    pixels = Pixelise(constit, eta_edges, phi_edges, cutoff)
    pixels = RotateJet(pixels, jets)
    pixels = ReflectJet(pixels, jets)
    if normalise:
        pixels = NormaliseJet(pixels)
    return pixels
