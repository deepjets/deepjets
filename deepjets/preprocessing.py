import numpy as np
from skimage.transform import rotate as rotate_pixels


def translate(jet_csts, all_jets):
    """Translate constituents and jets, leading subjet at (eta, phi) = (0, 0).
    """
    # Translate constituents
    jet_csts['eta'] -= all_jets['eta'][1]
    jet_csts['phi'] -= all_jets['phi'][1]
    # Ensure phi in [-pi, pi]
    jet_csts['phi'] = np.mod(jet_csts['phi'] + np.pi, 2*np.pi) - np.pi

    # Translate jets
    all_jets['eta'] -= all_jets['eta'][1]
    all_jets['phi'] -= all_jets['phi'][1]
    # Ensure phi in [-pi, pi]
    all_jets['phi'] = np.mod(all_jets['phi'] + np.pi, 2*np.pi) - np.pi


def pixelize(jet_csts, eta_edges, phi_edges, cutoff=0.0):
    """Return eta-phi histogram of transverse energy deposits.

    Optionally set all instensities below cutoff to zero.
    """
    pixels, _, _ = np.histogram2d(
        jet_csts['phi'], jet_csts['eta'],
        bins=(phi_edges, eta_edges),
        weights=jet_csts['ET'] * (jet_csts['ET'] > cutoff))
    return pixels


def rotate(pixels, all_jets):
    """Return rotated and repixelised image array.

    Rotation puts subleading subjet or first principle component at -pi/2.
    Repixelisation interpolates with cubic spline.
    """
    if len(all_jets) > 2:
        # Use subleading subject information to rotate
        theta = np.arctan2(all_jets['phi'][2], all_jets['eta'][2])
        theta = 90.0 + theta * 180.0 / np.pi
        return rotate_pixels(pixels, theta, order=3)

    # Use principle component of image intensity to rotate
    width, height  = pixels.shape
    pix_coords     = np.array([[i, j] for j in range(-height+1, height, 2)
                                      for i in range(-width+1, width, 2)])
    covX           = np.cov(pix_coords, rowvar=0,
                            aweights=np.reshape(pixels, (width*height)),
                            bias=1)
    e_vals, e_vecs = np.linalg.eigh(covX)
    pc             = e_vecs[:, -1]
    theta          = np.arctan2(pc[1], pc[0])
    theta          = 90 + theta * 180.0 / np.pi
    t_pixels       = rotate_pixels(pixels, theta, order=3)
    # Check orientation of principle component
    pix_bot = np.sum(t_pixels[:-(-height//2)].flatten())
    pix_top = np.sum(t_pixels[(height//2):].flatten())

    if pix_top > pix_bot:
        t_pixels = rotate_pixels(t_pixels, 180.0, order=3)
        theta    = theta + 180.0

    return t_pixels


def reflect(pixels, all_jets):
    """Return reflected image array.

    Reflection puts subsubleading subjet or highest intensity on right side.
    """

    width, height = pixels.shape

    if len(all_jets) > 3:
        # Use subsubleading subject information to reflect
        theta  = np.arctan2(all_jets['phi'][2], all_jets['eta'][2])
        theta  = (np.pi/2)+theta
        parity = np.sign(np.cos(-theta)*all_jets['eta'][3] -
                         np.sin(-theta)*all_jets['phi'][3])
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


def normalize(pixels):
    """Return normalized image array: sum(I**2) == 1.
    """
    return pixels / np.sum(pixels**2)


def preprocess(jets, constit, eta_edges, phi_edges,
               cutoff=0.1, norm=False):
    translate(constit, jets)
    pixels = pixelize(constit, eta_edges, phi_edges, cutoff)
    pixels = rotate(pixels, jets)
    pixels = reflect(pixels, jets)
    if norm:
        pixels = normalize(pixels)
    return pixels
