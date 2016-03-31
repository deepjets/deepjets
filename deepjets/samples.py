from joblib import Parallel, delayed
import h5py
import numpy as np
from numpy.lib.recfunctions import append_fields

from .generate import generate, get_generator_input
from .preprocessing import preprocess, pixel_edges


def get_images(
        generator_params, nevents, pt_min, pt_max,
        pix_size=(0.1, 0.1), image_size=25, normalize=True, jet_size=1.0,
        subjet_size_fraction=0.5, zoom=True, **kwargs):
    """
    Return image array and weights
    """
    gen_input = get_generator_input('pythia', **generator_params)

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
    while ievent < nevents:
        for event in generate(gen_input, nevents, jet_size=jet_size,
                              subjet_size_fraction=subjet_size_fraction,
                              trimmed_pt_min=pt_min, trimmed_pt_max=pt_max,
                              compute_auxvars=True,
                              **kwargs):
            image = preprocess(event.subjets, event.trimmed_constit, edges,
                               zoom=1. / event.shrinkage if zoom else False,
                               normalize=normalize,
                               out_width=image_size)
            images[ievent] = image
            pt[ievent] = event.jets[0]['pT']
            pt_trimmed[ievent] = event.jets[1]['pT']
            mass[ievent] = event.jets[0]['mass']
            mass_trimmed[ievent] = event.jets[1]['mass']
            subjet_dr[ievent] = event.subjet_dr
            tau_1[ievent] = event.tau_1
            tau_2[ievent] = event.tau_2
            tau_3[ievent] = event.tau_3
            ievent += 1
            if ievent == nevents:
                break

    auxvars = np.core.records.fromarrays(
        [pt, pt_trimmed, mass, mass_trimmed, subjet_dr, tau_1, tau_2, tau_3],
        names='pt,pt_trimmed,mass,mass_trimmed,subjet_dr,tau_1,tau_2,tau_3')

    return images, auxvars


def _generate_one_bin(
        generator_params, nevents_per_pt_bin, pt_lo, pt_hi, **kwargs):
    params_dict = {
        'PhaseSpace:pTHatMin': pt_lo - 20.,
        'PhaseSpace:pTHatMax': pt_hi + 20.}
    # defensive copy
    generator_params = generator_params.copy()
    generator_params['params_dict'] = params_dict
    return get_images(
        generator_params, nevents_per_pt_bin, pt_min=pt_lo, pt_max=pt_hi,
        **kwargs)


def get_weights(pt, pt_min, pt_max, pt_bins):
    # Compute weights such that pT distribution is flat
    pt_hist, edges = np.histogram(
        pt, bins=np.linspace(pt_min, pt_max, pt_bins + 1))
    # Normalize
    pt_hist = np.true_divide(pt_hist, pt_hist.sum())
    image_weights = np.true_divide(
        1., np.take(pt_hist, np.searchsorted(edges, pt) - 1))
    image_weights = np.true_divide(image_weights, image_weights.mean())
    return image_weights


def get_sample(
        generator_params, nevents_per_pt_bin, pt_min, pt_max, pt_bins=10,
        n_jobs=-1, **kwargs):
    """
    Construct a sample of images over a pT range by combining samples
    constructed in pT intervals in this range.
    """
    random_state = kwargs.get('random_state', None)
    pt_bin_edges = np.linspace(pt_min, pt_max, pt_bins + 1)

    out = Parallel(n_jobs=n_jobs)(
        delayed(_generate_one_bin)(
            generator_params, nevents_per_pt_bin, pt_lo, pt_hi, **kwargs)
        for pt_lo, pt_hi in zip(pt_bin_edges[:-1], pt_bin_edges[1:]))

    images = np.concatenate([x[0] for x in out])
    auxvars = np.concatenate([x[1] for x in out])
    pt = auxvars['pt_trimmed']

    image_weights = get_weights(pt, pt_min, pt_max, pt_bins * 4)

    # add weights column to auxvars
    auxvars = append_fields(auxvars, 'weights', data=image_weights)

    # shuffle
    random_state = np.random.RandomState(generator_params.get('random_state', 0))
    permute_idx = random_state.permutation(images.shape[0])
    images = images[permute_idx]
    auxvars = auxvars[permute_idx]

    return images, auxvars


def make_flat_sample(filename, pt_min, pt_max, pt_bins=20):
    """ Crop and weight a dataset such that pt is within pt_min and pt_max
    and the pt distribution is approximately flat. Return the images and
    weights.
    """
    with h5py.File(filename, 'r') as hfile:
        images = hfile['images']
        auxvars = hfile['auxvars']
        jet_pt_accept = ((auxvars['pt_trimmed'] >= pt_min) &
                         (auxvars['pt_trimmed'] < pt_max))
        images = np.take(images, np.where(jet_pt_accept)[0], axis=0)
        jet_pt = auxvars['pt_trimmed'][jet_pt_accept]
    weights = get_weights(jet_pt, pt_min, pt_max, pt_bins)
    return images, weights


def dataset_append(filename, datasetname, data):
    """ Append an array to an HDF5 dataset
    """
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
