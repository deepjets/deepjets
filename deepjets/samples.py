from joblib import Parallel, delayed
import h5py
import numpy as np
from numpy.lib.recfunctions import append_fields

from .generate import generate, get_generator_input
from .preprocessing import preprocess, pixel_edges

DTYPE = np.double
dt_jet = np.dtype([('pT', DTYPE), ('eta', DTYPE), ('phi', DTYPE), ('mass', DTYPE)])
dt_jets = h5py.special_dtype(vlen=dt_jet)
dt_constit = h5py.special_dtype(vlen=np.dtype([('ET', DTYPE), ('eta', DTYPE), ('phi', DTYPE)]))


def create_event_datasets(h5file, events, jet_size, subjet_size_fraction):
    h5file.create_dataset('jet', (events,), maxshape=(events,), dtype=dt_jet, chunks=True)
    h5file.create_dataset('trimmed_jet', (events,), maxshape=(events,), dtype=dt_jet, chunks=True)
    h5file.create_dataset('subjets', (events,), maxshape=(events,), dtype=dt_jets, chunks=True)
    h5file.create_dataset('constituents', (events,), maxshape=(events,), dtype=dt_constit, chunks=True)
    h5file.create_dataset('trimmed_constituents', (events,), maxshape=(events,), dtype=dt_constit, chunks=True)
    h5file.create_dataset('shrinkage', (events,), maxshape=(events,), dtype=DTYPE, chunks=True)
    h5file.create_dataset('subjet_dr', (events,), maxshape=(events,), dtype=DTYPE, chunks=True)
    h5file.create_dataset('tau_1', (events,), maxshape=(events,), dtype=DTYPE, chunks=True)
    h5file.create_dataset('tau_2', (events,), maxshape=(events,), dtype=DTYPE, chunks=True)
    h5file.create_dataset('tau_3', (events,), maxshape=(events,), dtype=DTYPE, chunks=True)

    # metadatasets
    dset_jet_size = h5file.create_dataset('jet_size', (1,), dtype=DTYPE)
    dset_subjet_size_fraction = h5file.create_dataset('subjet_size_fraction', (1,), dtype=DTYPE)
    dset_jet_size[0] = jet_size
    dset_subjet_size_fraction[0] = subjet_size_fraction


def get_images(generator_params, nevents, pt_min, pt_max,
               pix_size=(0.1, 0.1), image_size=25, normalize=True,
               jet_size=1.0, subjet_size_fraction=0.5, zoom=True, **kwargs):
    """
    Return image array and weights
    """
    params_dict = {
        'PhaseSpace:pTHatMin': pt_min - 20.,
        'PhaseSpace:pTHatMax': pt_max + 20.}
    # defensive copy
    generator_params = generator_params.copy()
    generator_params['params_dict'] = params_dict

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

    auxvars = np.core.records.fromarrays(
        [pt, pt_trimmed, mass, mass_trimmed, subjet_dr, tau_1, tau_2, tau_3],
        names='pt,pt_trimmed,mass,mass_trimmed,subjet_dr,tau_1,tau_2,tau_3')

    return images, auxvars


def get_events(h5file, generator_params, nevents, pt_min, pt_max, **kwargs):
    params_dict = {
        'PhaseSpace:pTHatMin': pt_min - 20.,
        'PhaseSpace:pTHatMax': pt_max + 20.}
    # defensive copy
    generator_params = generator_params.copy()
    generator_params['params_dict'] = params_dict

    gen_input = get_generator_input('pythia', **generator_params)

    dset_jet = h5file['jet']
    dset_trimmed_jet = h5file['trimmed_jet']
    dset_subjets = h5file['subjets']
    dset_constit = h5file['constituents']
    dset_trimmed_constit = h5file['trimmed_constituents']
    dset_shrinkage = h5file['shrinkage']
    dset_dr_subjets = h5file['subjet_dr']
    dset_tau_1 = h5file['tau_1']
    dset_tau_2 = h5file['tau_2']
    dset_tau_3 = h5file['tau_3']

    ievent = 0
    for event in generate(gen_input, nevents,
                          trimmed_pt_min=pt_min,
                          trimmed_pt_max=pt_max,
                          compute_auxvars=True,
                          **kwargs):
        dset_jet[ievent] = event.jets[0]
        dset_trimmed_jet[ievent] = event.jets[1]
        dset_subjets[ievent] = event.subjets
        dset_constit[ievent] = event.constit
        dset_trimmed_constit[ievent] = event.trimmed_constit
        dset_shrinkage[ievent] = event.shrinkage
        dset_dr_subjets[ievent] = event.subjet_dr
        dset_tau_1[ievent] = event.tau_1
        dset_tau_2[ievent] = event.tau_2
        dset_tau_3[ievent] = event.tau_3
        ievent += 1


def get_flat_weights(pt, pt_min, pt_max, pt_bins):
    # Compute weights such that pT distribution is flat
    pt_hist, edges = np.histogram(
        pt, bins=np.linspace(pt_min, pt_max, pt_bins + 1))
    # Normalize
    pt_hist = np.true_divide(pt_hist, pt_hist.sum())
    image_weights = np.true_divide(
        1., np.take(pt_hist, np.searchsorted(edges, pt) - 1))
    image_weights = np.true_divide(image_weights, image_weights.mean())
    return image_weights


def get_flat_images(generator_params, nevents_per_pt_bin,
                    pt_min, pt_max, pt_bins=10,
                    n_jobs=-1, **kwargs):
    """
    Construct a sample of images over a pT range by combining samples
    constructed in pT intervals in this range.
    """
    random_state = kwargs.get('random_state', None)
    pt_bin_edges = np.linspace(pt_min, pt_max, pt_bins + 1)

    out = Parallel(n_jobs=n_jobs)(
        delayed(get_images)(
            generator_params, nevents_per_pt_bin, pt_lo, pt_hi, **kwargs)
        for pt_lo, pt_hi in zip(pt_bin_edges[:-1], pt_bin_edges[1:]))

    images = np.concatenate([x[0] for x in out])
    auxvars = np.concatenate([x[1] for x in out])
    pt = auxvars['pt_trimmed']

    image_weights = get_flat_weights(pt, pt_min, pt_max, pt_bins * 4)

    # add weights column to auxvars
    auxvars = append_fields(auxvars, 'weights', data=image_weights)

    # shuffle
    random_state = np.random.RandomState(generator_params.get('random_state', 0))
    permute_idx = random_state.permutation(images.shape[0])
    images = images[permute_idx]
    auxvars = auxvars[permute_idx]

    return images, auxvars


def make_flat_images(filename, pt_min, pt_max, pt_bins=20):
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
    weights = get_flat_weights(jet_pt, pt_min, pt_max, pt_bins)
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


def get_flat_events(h5file, generator_params, nevents_per_pt_bin,
                    pt_min, pt_max, pt_bins=10, **kwargs):
    """
    Construct a sample of events over a pT range by combining samples
    constructed in pT intervals in this range.
    """
    pt_bin_edges = np.linspace(pt_min, pt_max, pt_bins + 1)

    for pt_lo, pt_hi in zip(pt_bin_edges[:-1], pt_bin_edges[1:]):
        get_events(h5file, generator_params, nevents_per_pt_bin,
                   pt_lo, pt_hi, **kwargs)

    pt = h5file['trimmed_jet']['pT']
    event_weights = get_flat_weights(pt, pt_min, pt_max, pt_bins * 4)
    h5file.create_dataset('weights', data=event_weights)
