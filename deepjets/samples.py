from joblib import Parallel, delayed
import h5py
import numpy as np
from numpy.lib.recfunctions import append_fields

from .generate import generate_events, get_generator_input
from .preprocessing import preprocess, pixel_edges

DTYPE = np.double
dt_jet = np.dtype(
    [('pT', DTYPE), ('eta', DTYPE), ('phi', DTYPE), ('mass', DTYPE)])
dt_jets = h5py.special_dtype(vlen=dt_jet)
dt_constit = h5py.special_dtype(vlen=np.dtype(
    [('ET', DTYPE), ('eta', DTYPE), ('phi', DTYPE)]))
dt_particle = np.dtype(
    [('E', DTYPE), ('px', DTYPE), ('py', DTYPE), ('pz', DTYPE), ('mass', DTYPE),
     ('prodx', DTYPE), ('prody', DTYPE), ('prodz', DTYPE), ('prodt', DTYPE),
     ('pdgid', DTYPE)])  # extra info needed by Delphes
dt_particles = h5py.special_dtype(vlen=dt_particle)
dt_candidate = np.dtype(
    [('E', DTYPE), ('px', DTYPE), ('py', DTYPE), ('pz', DTYPE)])
dt_candidates = h5py.special_dtype(vlen=dt_candidate)


def create_jets_datasets(h5file, events, jet_size, subjet_size_fraction):
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


def create_event_datasets(h5file, events, delphes=False, nweights=0):
    dtype = dt_candidates if delphes else dt_particles
    h5file.create_dataset('events', (events,), maxshape=(events,),
                          dtype=dtype, chunks=True)
    if nweights > 0:
        h5file.create_dataset('weights', (events, nweights),
                              maxshape=(events, nweights),
                              dtype=DTYPE, chunks=True)


def get_images(generator_params, nevents, pt_min, pt_max,
               pixel_size=(0.1, 0.1), image_size=25, normalize=True,
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
        pixel_size=pixel_size)

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


def get_events(h5file, generator_params, nevents, pt_min, pt_max,
               offset=0, **kwargs):
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

    ievent = offset
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


def make_flat_images(filename, pt_min, pt_max, pt_bins=20,
                     mass_min=None, mass_max=None):
    """ Crop and weight a dataset such that pt is within pt_min and pt_max
    and the pt distribution is approximately flat. Return the images and
    weights.
    """
    with h5py.File(filename, 'r') as hfile:
        images = hfile['images']
        auxvars = hfile['auxvars']
        accept = ((auxvars['pt_trimmed'] >= pt_min) &
                  (auxvars['pt_trimmed'] < pt_max))
        if mass_min is not None:
            accept &= auxvars['mass_trimmed'] >= mass_min
        if mass_max is not None:
            accept &= auxvars['mass_trimmed'] < mass_max
        images = np.take(images, np.where(accept)[0], axis=0)
        jet_pt = auxvars['pt_trimmed'][accept]
        auxvars = auxvars[accept]
    weights = get_flat_weights(jet_pt, pt_min, pt_max, pt_bins)
    return images, auxvars, weights


def dataset_append(h5output, datasetname, data,
                   dtype=None, chunked_read=False,
                   selection=None, indices=None):
    """ Append an array to an HDF5 dataset
    """
    if indices is not None and selection is not None:
        raise NotImplementedError(
            "handling both selection and indices is not implemented")
    if isinstance(h5output, basestring):
        h5file = h5py.File(h5output, 'a')
        own_file = True
    else:
        own_file = False
        h5file = h5output
    if dtype is None:
        dtype = data.dtype
        convert_type = False
    else:
        convert_type = True
    if datasetname not in h5file:
        dset = h5file.create_dataset(
            datasetname, data.shape,
            maxshape=[None,] + list(data.shape)[1:],
            dtype=dtype, chunks=True)
        prev_size = 0
    else:
        dset = h5file[datasetname]
        prev_size = dset.shape[0]
    if selection is not None:
        dset.resize(prev_size + selection.sum(), axis=0)
    elif indices is not None:
        dset.resize(prev_size + indices.shape[0], axis=0)
    else:
        dset.resize(prev_size + data.shape[0], axis=0)
    if chunked_read and isinstance(data, h5py.Dataset):
        # read dataset in chunks of size chunked_read
        if len(data.shape) > 1:
            elem_size = np.prod(data.shape[1:]) * data.dtype.itemsize
        else:
            elem_size = data.dtype.itemsize
        chunk_size = int(chunked_read) / int(elem_size)
        if chunk_size == 0:
            raise RuntimeError(
                "chunked_read is smaller than a single "
                "element along first axis of input")
        start = 0
        offset = prev_size
        if indices is not None:
            end = indices.shape[0]
        else:
            end = len(data)
        while start < end:
            stop = min(end, start + chunk_size)
            if indices is not None:
                indices_chunk = indices[start:stop]
                # index list must be sorted in h5py
                indices_argsort = np.argsort(indices_chunk)
                data_chunk = data[(indices_chunk[indices_argsort]).tolist()]
                # unsort
                data_chunk = np.take(data_chunk, indices_argsort, axis=0)
            else:
                data_chunk = data[start:stop]
                if selection is not None:
                    data_chunk = np.take(data_chunk,
                                         np.where(selection[start:stop]),
                                         axis=0)[0]
            if convert_type:
                data_chunk = data_chunk.astype(dtype)
            dset[offset:offset + data_chunk.shape[0]] = data_chunk
            start = stop
            offset += data_chunk.shape[0]
    else:
        if selection is not None:
            data = np.take(data, np.where(selection), axis=0)[0]
        elif indices is not None:
            data = np.take(data, indices, axis=0)
        if convert_type:
            data = data.astype(dtype)
        dset[prev_size:] = data
    if own_file:
        h5file.close()
    else:
        h5file.flush()


def get_flat_events(h5file, generator_params, nevents_per_pt_bin,
                    pt_min, pt_max, pt_bins=10, **kwargs):
    """
    Construct a sample of events over a pT range by combining samples
    constructed in pT intervals in this range.
    """
    pt_bin_edges = np.linspace(pt_min, pt_max, pt_bins + 1)

    offset = 0
    for pt_lo, pt_hi in zip(pt_bin_edges[:-1], pt_bin_edges[1:]):
        get_events(h5file, generator_params, nevents_per_pt_bin,
                   pt_lo, pt_hi, offset=offset, **kwargs)
        offset += nevents_per_pt_bin

    pt = h5file['trimmed_jet']['pT']
    event_weights = get_flat_weights(pt, pt_min, pt_max, pt_bins * 4)
    h5file.create_dataset('weights', data=event_weights)
