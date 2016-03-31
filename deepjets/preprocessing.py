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


def pixel_edges(
        jet_size=1.2, subjet_size_fraction=0.5, pix_size=(0.1,0.1),
        border_size=1):
    """Return pixel edges required to contain all subjets.
    """
    im_edge  = (1+(1+border_size)*subjet_size_fraction)*jet_size
    return (np.arange(-im_edge, im_edge+pix_size[0], pix_size[0]),
            np.arange(-im_edge, im_edge+pix_size[1], pix_size[1]))


def pixelize(jet_csts, edges, cutoff=0.1):
    """Return eta-phi histogram of transverse energy deposits.

    Optionally set all instensities below cutoff to zero.
    """
    image, _, _ = np.histogram2d(
        jet_csts['eta'], jet_csts['phi'],
        bins=(edges[0], edges[1]),
        weights=jet_csts['ET'] * (jet_csts['ET'] > cutoff))
    return image


def rotate_image(image, subjets):
    """Return rotated and repixelised image array.

    Rotation puts subleading subjet or first principle component at -pi/2.
    Repixelisation interpolates with cubic spline.
    """
    # Use subleading subject information to rotate
    if len(subjets) > 1:
        theta = np.arctan2(subjets['phi'][1], subjets['eta'][1])
        theta = -90.0-(theta*180.0/np.pi)
        return transform.rotate(image, theta, order=3)

    # Use principle component of image intensity to rotate
    width, height = image.shape
    pix_coords = np.array([[i, j] for i in range(-width+1, width, 2)
                           for j in range(-height+1, height, 2)])
    covX = np.cov(pix_coords, aweights=np.reshape(image, (width*height)),
                  rowvar=0, bias=1)
    e_vals, e_vecs = np.linalg.eigh(covX)
    pc = e_vecs[:,-1]
    theta = np.arctan2(pc[1], pc[0])
    theta = -90.0-(theta*180.0/np.pi)
    t_image = transform.rotate(image, theta, order=3)
    # Check orientation of principle component
    pix_bot = np.sum(t_image[:, :-(-height//2)])
    pix_top = np.sum(t_image[:, (height//2):])
    if pix_top > pix_bot:
        t_image = transform.rotate(t_image, 180.0, order=3)
        theta += 180.0
    return t_image


def reflect_image(image, subjets):
    """Return reflected image array.

    Reflection puts subsubleading subjet or highest intensity on right side.
    """
    width, height = image.shape
    if len(subjets) > 2:
        # Use subsubleading subject information to find parity
        theta = np.arctan2(subjets['phi'][1], subjets['eta'][1])
        theta = -(np.pi/2)-theta
        parity = np.sign(np.cos(-theta)*subjets['eta'][2] +
                         np.sin(-theta)*subjets['phi'][2])
    else:
        # Use intensity to find parity
        pix_l = np.sum(image[:-(-width//2)].flatten())
        pix_r = np.sum(image[(width//2):].flatten())
        parity = np.sign(pix_r - pix_l)

    if parity >= 0:
        return image
    t_image = np.array(image)
    for i in range(width):
        t_image[i] = image[-i-1]
    return t_image


def zoom_image_fixed_size(image, zoom):
    """Return rescaled and cropped image array.

    Expansion interpolates with cubic spline.
    """
    if zoom < 1:
        raise ValueError("Zoom scale factor must be at least 1.")
    elif zoom == 1:
        # copy
        return np.array(image)

    width, height = image.shape
    t_width = int(np.ceil(zoom*width))
    t_height = int(np.ceil(zoom*height))
    if t_width//2 != width//2:
        t_width -= 1
    if t_height//2 != height//2:
        t_height -= 1
    t_image = transform.resize(image, (t_width, t_height), order=3)
    return t_image[(t_width-width)/2:(t_width+width)/2,
                   (t_height-height)/2:(t_height+height)/2]


def zoom_image(image, zoom, out_width=25):
    """Return rescaled and cropped image array with width out_width.

    Expansion interpolates with cubic spline.
    """
    if zoom < 1:
        raise ValueError("Zoom scale factor must be at least 1.")

    width, height = image.shape
    out_height = int(np.rint(float(out_width*height)/width))
    t_width = int(np.rint(out_width*zoom))
    t_height = int(np.rint(out_height*zoom))
    if t_width//2 != out_width//2:
        t_width += 1
    if t_height//2 != out_height//2:
        t_height += 1
    t_image = transform.resize(image, (t_width, t_height), order=3)
    return t_image[(t_width-out_width)/2:(t_width+out_width)/2,
                   (t_height-out_height)/2:(t_height+out_height)/2]


def normalize_image(image):
    """Return normalized image array: sum(I**2) == 1.
    """
    return image / np.sum(image**2)


def preprocess_fixed_size(subjets, constit, edges,
                          cutoff=0.1,
                          rotate=True,
                          reflect=True,
                          zoom=False,
                          normalize=False):
    translate(constit, subjets)
    image = pixelize(constit, edges, cutoff)
    if rotate:
        image = rotate_image(image, subjets)
    if reflect:
        image = reflect_image(image, subjets)
    if zoom is not False:
        image = zoom_image_fixed_size(image, zoom)
    if normalize:
        image = normalize_image(image)
    return image


def preprocess(subjets, constits, edges,
               cutoff=0.1,
               rotate=True,
               reflect=True,
               zoom=False,
               out_width=25,
               normalize=False):
    translate(constits, subjets)
    image = pixelize(constits, edges)
    if rotate:
        image = rotate_image(image, subjets)
    if reflect:
        image = reflect_image(image, subjets)
    if zoom is not False:
        image = zoom_image(image, zoom, out_width)
    else:
        image = zoom_image(image, 1., out_width)
    if normalize:
        image = normalize_image(image)
    return image
