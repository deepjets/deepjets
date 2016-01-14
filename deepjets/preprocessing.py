import numpy as np
from skimage import transform


def translate(jet_csts, subjets):
    """Translate constituents and jets, leading subjet at (eta, phi) = (0, 0).
    """
    # Translate constituents
    jet_csts['eta'] -= subjets['eta'][0]
    jet_csts['phi'] -= subjets['phi'][0]
    # Ensure phi in [-pi, pi]
    jet_csts['phi'] = np.mod(jet_csts['phi'] + np.pi, 2*np.pi) - np.pi
    # Translate jets
    subjets['eta'] -= subjets['eta'][0]
    subjets['phi'] -= subjets['phi'][0]
    # Ensure phi in [-pi, pi]
    subjets['phi'] = np.mod(subjets['phi'] + np.pi, 2*np.pi) - np.pi


def pixel_edges(jet_size=1.2, subjet_size_fraction=0.5, pix_size=(0.1,0.1), border_size=2):
    """Return pixel edges required to contain all subjets.
    """
    im_edge  = (1+(1+border_size)*subjet_size_fraction)*jet_size
    return (np.arange(-im_edge, im_edge+pix_size[0], pix_size[0]),
            np.arange(-im_edge, im_edge+pix_size[1], pix_size[1]))


def pixelize(jet_csts, edges, cutoff=0.1):
    """Return eta-phi histogram of transverse energy deposits.

    Optionally set all instensities below cutoff to zero.
    """
    pixels, _, _ = np.histogram2d(
        jet_csts['eta'], jet_csts['phi'],
        bins=(edges[0], edges[1]),
        weights=jet_csts['ET'] * (jet_csts['ET'] > cutoff))
    return pixels


def rotate_image(pixels, subjets):
    """Return rotated and repixelised image array.

    Rotation puts subleading subjet or first principle component at -pi/2.
    Repixelisation interpolates with cubic spline.
    """
    # Use subleading subject information to rotate
    if len(subjets) > 1:
        theta = np.arctan2(subjets['phi'][1], subjets['eta'][1])
        theta = -90.0-(theta*180.0/np.pi)
        return transform.rotate(pixels, theta, order=3)

    # Use principle component of image intensity to rotate
    width, height = pixels.shape
    pix_coords = np.array([[i, j] for i in range(-width+1, width, 2)
                           for j in range(-height+1, height, 2)])
    covX = np.cov(pix_coords, aweights=np.reshape(pixels, (width*height)),
                  rowvar=0, bias=1)
    e_vals, e_vecs = np.linalg.eigh(covX)
    pc = e_vecs[:,-1]
    theta = np.arctan2(pc[1], pc[0])
    theta = -90.0-(theta*180.0/np.pi)
    t_pixels = transform.rotate(pixels, theta, order=3)
    # Check orientation of principle component
    pix_bot = np.sum(t_pixels[:, :-(-height//2)])
    pix_top = np.sum(t_pixels[:, (height//2):])
    if pix_top > pix_bot:
        t_pixels = transform.rotate(t_pixels, 180.0, order=3)
        theta += 180.0
    return t_pixels


def reflect_image(pixels, subjets):
    """Return reflected image array.

    Reflection puts subsubleading subjet or highest intensity on right side.
    """
    width, height = pixels.shape
    if len(subjets) > 2:
        # Use subsubleading subject information to find parity
        theta = np.arctan2(subjets['phi'][1], subjets['eta'][1])
        theta = -(np.pi/2)-theta
        parity = np.sign(np.cos(-theta)*subjets['eta'][2] +
                         np.sin(-theta)*subjets['phi'][2])
    else:
        # Use intensity to find parity
        pix_l = np.sum(pixels[:-(-width//2)].flatten())
        pix_r = np.sum(pixels[(width//2):].flatten())
        parity = np.sign(pix_r - pix_l)

    if parity >= 0:
        return pixels
    t_pixels = np.array(pixels)
    for i in range(width):
        t_pixels[i] = pixels[-i-1]
    return t_pixels


def zoom_image_fixed_size(pixels, zoom):
    """Return rescaled and cropped image array.

    Expansion interpolates with cubic spline.
    """
    if zoom < 1:
        raise ValueError("zoom scale factor must be at least 1")
    elif zoom == 1:
        # copy
        return np.array(pixels)

    width, height = pixels.shape
    t_width = int(np.ceil(zoom*width))
    t_height = int(np.ceil(zoom*height))
    if t_width//2 != width//2:
        t_width -= 1
    if t_height//2 != height//2:
        t_height -= 1
    t_pixels = transform.resize(pixels, (t_width, t_height), order=3)
    return t_pixels[(t_width-width)/2:(t_width+width)/2,
                    (t_height-height)/2:(t_height+height)/2]


def zoom_image(pixels, zoom, out_width=25):
    """Return rescaled and cropped image array with width out_width.

    Expansion interpolates with cubic spline.
    """
    if zoom < 1:
        raise ValueError("zoom scale factor must be at least 1")

    width, height = pixels.shape
    out_height = int(np.rint(float(out_width*height)/width))
    t_width = int(np.rint(out_width*zoom))
    t_height = int(np.rint(out_height*zoom))
    if t_width//2 != out_width//2:
        t_width += 1
    if t_height//2 != out_height//2:
        t_height += 1
    t_pixels = transform.resize(pixels, (t_width, t_height), order=3)
    return t_pixels[(t_width-out_width)/2:(t_width+out_width)/2,
                    (t_height-out_height)/2:(t_height+out_height)/2]


def normalize_image(pixels):
    """Return normalized image array: sum(I**2) == 1.
    """
    return pixels / np.sum(pixels**2)


def preprocess_fixed_size(subjets, constit, edges,
                          cutoff=0.1,
                          rotate=True,
                          reflect=True,
                          zoom=False,
                          normalize=False):
    translate(constit, subjets)
    pixels = pixelize(constit, edges, cutoff)
    if rotate:
        pixels = rotate_image(pixels, subjets)
    if reflect:
        pixels = reflect_image(pixels, subjets)
    if zoom is not False:
        pixels = zoom_image_fixed_size(pixels, zoom)
    if normalize:
        pixels = normalize_image(pixels)
    return pixels


def preprocess(subjets, constits, edges,
               cutoff=0.1,
               rotate=True,
               reflect=True,
               zoom=False,
               out_width=25,
               normalize=False):
    translate(constits, subjets)
    pixels = pixelize(constits, edges)
    if rotate:
        pixels = rotate_image(pixels, subjets)
    if reflect:
        pixels = reflect_image(pixels, subjets)
    if zoom is not False:
        pixels = zoom_image(pixels, zoom, out_width)
    if normalize:
        pixels = normalize_image(pixels)
    return pixels
