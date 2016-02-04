from joblib import Parallel, delayed

from .generate import generate
from .preprocessing import preprocess, pixel_edges
from .utils import pT, tot_mom

import numpy as np


def get_images(config, nevents, pt_min, pt_max,
               pix_size=(0.1, 0.1), image_size=25,
               normalize=True, jet_size=1.0, subjet_size_fraction=0.5,
               **kwargs):
    """
    Return image array and weights
    """
    images = np.empty((nevents, image_size, image_size), dtype=np.double)
    pt = np.empty(nevents, dtype=np.double)
    edges = pixel_edges(
        jet_size=jet_size,
        subjet_size_fraction=subjet_size_fraction,
        pix_size=pix_size)

    ievent = 0
    for event in generate(config, nevents, jet_size=jet_size,
                          subjet_size_fraction=subjet_size_fraction,
                          trimmed_pt_min=pt_min, trimmed_pt_max=pt_max,
                          **kwargs):
        jets, subjets, constit, trimmed_constit, shrinkage = event
        image = preprocess(subjets, trimmed_constit, edges,
                           zoom=1. / shrinkage,
                           normalize=normalize,
                           out_width=image_size)
        images[ievent] = image
        # trimmed pT
        pt[ievent] = jets[1]['pT']
        ievent += 1
    return images, pt


def _generate_one_bin(config, nevents_per_pt_bin, pt_lo, pt_hi, **kwargs):
    params_dict = {
        'PhaseSpace:pTHatMin': pt_lo - 20.,
        'PhaseSpace:pTHatMax': pt_hi + 20.}
    return get_images(config, nevents_per_pt_bin,
                      pt_min=pt_lo, pt_max=pt_hi,
                      params_dict=params_dict,
                      **kwargs)


def get_sample(config, nevents_per_pt_bin, pt_min, pt_max,
               pt_bins=10, n_jobs=-1, **kwargs):
    """
    Construct a sample of images over a pT range by combining samples
    constructed in pT intervals in this range.
    """
    pt_bin_edges = np.linspace(pt_min, pt_max, pt_bins + 1)

    out = Parallel(n_jobs=n_jobs)(
        delayed(_generate_one_bin)(config, nevents_per_pt_bin, pt_lo, pt_hi, **kwargs)
            for pt_lo, pt_hi in zip(pt_bin_edges[:-1], pt_bin_edges[1:]))

    images = np.concatenate([x[0] for x in out])
    pt = np.concatenate([x[1] for x in out])

    # Compute weights such that pT distribution is flat
    pt_hist, edges = np.histogram(pt, bins=np.linspace(pt_min, pt_max, (pt_bins * 4) + 1))
    # Normalize
    pt_hist = np.true_divide(pt_hist, pt_hist.sum())
    image_weights = np.true_divide(1., np.take(pt_hist, np.searchsorted(edges, pt) - 1))
    image_weights = np.true_divide(image_weights, image_weights.mean())

    return images, pt, image_weights
