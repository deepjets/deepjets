from joblib import Parallel, delayed
import h5py
import numpy as np
from numpy.lib.recfunctions import append_fields

from .generate import generate
from .preprocessing import preprocess, pixel_edges
from .utils import pT, tot_mom


def get_images(config, nevents, pt_min, pt_max,
               pix_size=(0.1, 0.1), image_size=25,
               normalize=True, jet_size=1.0, subjet_size_fraction=0.5,
               **kwargs):
    """
    Return image array and weights
    """
    images = np.empty((nevents, image_size, image_size), dtype=np.double)

    pt = np.empty(nevents, dtype=np.double)
    pt_trimmed = np.empty(nevents, dtype=np.double)
    mass = np.empty(nevents, dtype=np.double)
    mass_trimmed = np.empty(nevents, dtype=np.double)
    subjet_dr = np.empty(nevents, dtype=np.double)
    tau_1 = np.empty(nevents, dtype=np.double)
    tau_2 = np.empty(nevents, dtype=np.double)
    tau_3 = np.empty(nevents, dtype=np.double)

    edges = pixel_edges(
        jet_size=jet_size,
        subjet_size_fraction=subjet_size_fraction,
        pix_size=pix_size)

    ievent = 0
    for event in generate(config, nevents, jet_size=jet_size,
                          subjet_size_fraction=subjet_size_fraction,
                          trimmed_pt_min=pt_min, trimmed_pt_max=pt_max,
                          compute_auxvars=True,
                          **kwargs):
        jets, subjets, constit, trimmed_constit, shrinkage, auxvars = event
        image = preprocess(subjets, trimmed_constit, edges,
                           zoom=1. / shrinkage,
                           normalize=normalize,
                           out_width=image_size)
        images[ievent] = image
        pt[ievent] = jets[0]['pT']
        pt_trimmed[ievent] = jets[1]['pT']
        mass[ievent] = jets[0]['mass']
        mass_trimmed[ievent] = jets[1]['mass']
        subjet_dr[ievent] = auxvars['subjet_dr']
        tau_1[ievent] = auxvars['tau_1']
        tau_2[ievent] = auxvars['tau_2']
        tau_3[ievent] = auxvars['tau_3']
        ievent += 1

    auxvars = np.core.records.fromarrays(
        [pt, pt_trimmed, mass, mass_trimmed, subjet_dr, tau_1, tau_2, tau_3],
        names='pt,pt_trimmed,mass,mass_trimmed,subjet_dr,tau_1,tau_2,tau_3')

    return images, auxvars


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
    random_state = kwargs.get('random_state', None)
    pt_bin_edges = np.linspace(pt_min, pt_max, pt_bins + 1)

    out = Parallel(n_jobs=n_jobs)(
        delayed(_generate_one_bin)(config, nevents_per_pt_bin, pt_lo, pt_hi, **kwargs)
            for pt_lo, pt_hi in zip(pt_bin_edges[:-1], pt_bin_edges[1:]))

    images = np.concatenate([x[0] for x in out])
    auxvars = np.concatenate([x[1] for x in out])
    pt = auxvars['pt_trimmed']

    # Compute weights such that pT distribution is flat
    pt_hist, edges = np.histogram(pt, bins=np.linspace(pt_min, pt_max, (pt_bins * 4) + 1))
    # Normalize
    pt_hist = np.true_divide(pt_hist, pt_hist.sum())
    image_weights = np.true_divide(1., np.take(pt_hist, np.searchsorted(edges, pt) - 1))
    image_weights = np.true_divide(image_weights, image_weights.mean())

    # add weights column to auxvars
    auxvars = append_fields(auxvars, 'weights', data=image_weights)

    # shuffle
    random_state = np.random.RandomState(random_state)
    permute_idx = random_state.permutation(images.shape[0])
    images = images[permute_idx]
    auxvars = auxvars[permute_idx]

    return images, auxvars


def dataset_append(filename, datasetname, data):
    with h5py.File(filename, 'a') as h5file:
        if datasetname not in h5file:
            dset = h5file.create_dataset(
                datasetname, data.shape,
                maxshape=[None,] + list(data.shape)[1:],
                dtype=data.dtype)
            prev_size = 0
        else:
            dset = h5file[datasetname]
            prev_size = dset.shape[0]
            dset.resize(prev_size + data.shape[0], axis=0)
        dset[prev_size:] = data
