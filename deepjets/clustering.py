from ._libdeepjets import (cluster_mc, cluster_numpy,
                           cluster_hdf5, cluster_iterable)
from ._libdeepjets import MCInput
import h5py as h5
import inspect

__all__ = [
    'cluster',
    ]


def cluster(inputs,
            events=-1,
            skip_failed=True,
            eta_max=5.,
            jet_size=1.0, subjet_size=0.5,
            subjet_pt_min_fraction=0.05,
            subjet_dr_min=0.,
            trimmed_pt_min=-1., trimmed_pt_max=-1.,
            trimmed_mass_min=-1., trimmed_mass_max=-1.,
            shrink=False, shrink_mass=-1,
            compute_auxvars=False):
    """
    Cluster particles into jets. Inputs may be an MCInput, h5py Dataset,
    an array of particles (single event) or a generator that yields events
    of particles.

    The events and skip_failed arguments are only applied in the case that
    inputs is a generator function.
    """

    if jet_size <= 0:
        raise ValueError("jet_size must be greater than zero")

    if subjet_size <= 0 or subjet_size > 0.5 * jet_size:
        raise ValueError(
            "subjet_size must be in the range (0, 0.5 * jet_size]")

    kwargs = dict(
        eta_max=eta_max,
        jet_size=jet_size,
        subjet_size=subjet_size,
        subjet_pt_min_fraction=subjet_pt_min_fraction,
        subjet_dr_min=subjet_dr_min,
        trimmed_pt_min=trimmed_pt_min,
        trimmed_pt_max=trimmed_pt_max,
        trimmed_mass_min=trimmed_mass_min,
        trimmed_mass_max=trimmed_mass_max,
        shrink=shrink,
        shrink_mass=shrink_mass,
        compute_auxvars=compute_auxvars)

    if isinstance(inputs, MCInput):
        cluster_func = cluster_mc
    elif isinstance(inputs, h5.Dataset):
        cluster_func = cluster_hdf5
    else:
        cluster_func = cluster_iterable
        if inspect.isgenerator(inputs):
            kwargs['events'] = events
            kwargs['skip_failed'] = skip_failed
        else:
            # handle case where input is just one event
            inputs = [inputs]

    for event in cluster_func(inputs, **kwargs):
        yield event
