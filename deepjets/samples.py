from .generate import generate
from .preprocessing import preprocess, pixel_edges
from .utils import pT, tot_mom

import numpy as np


def get_images(config, nevents,
               pt_min=200., pt_max=500.,
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
                          subjet_size_fraction=subjet_size_fraction, **kwargs):
        jets, constit, trimmed_constit, shrinkage = event
        curr_pt = pT(*tot_mom(trimmed_constit))
        if not (pt_min < curr_pt <= pt_max):
            continue
        subjets = jets[1:]
        image = preprocess(subjets, trimmed_constit, edges,
                           zoom=1. / shrinkage,
                           normalize=normalize,
                           out_width=image_size)
        images[ievent] = image
        # trimmed pT
        pt[ievent] = pT(*tot_mom(trimmed_constit))
        ievent += 1
    return images, pt



def get_sample(config, nevents_per_pt_step, pt_steps=10, **kwargs):

    images, pt = get_images(config, nevents_per_pt_step, **kwargs)

    # WIP
