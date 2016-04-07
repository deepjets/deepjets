from ._libdeepjets import (cluster_hepmc, cluster_numpy,
                           cluster_hdf5, cluster_iterable)
from ._libdeepjets import HepMCInput
import numpy as np
import h5py as h5
import inspect

__all__ = [
    'cluster',
    ]


def cluster(inputs,
            events=-1,
            eta_max=5.,
            jet_size=1.0, subjet_size_fraction=0.5,
            subjet_pt_min_fraction=0.05,
            subjet_dr_min=0.,
            trimmed_pt_min=-1., trimmed_pt_max=-1.,
            trimmed_mass_min=-1., trimmed_mass_max=-1.,
            shrink=False, shrink_mass=-1,
            compute_auxvars=False,
            skip_failed=True):

    if subjet_size_fraction <= 0 or subjet_size_fraction > 0.5:
        raise ValueError("subjet_size_fraction must be in the range (0, 0.5]")

    kwargs = dict(
        eta_max=eta_max,
        jet_size=jet_size,
        subjet_size_fraction=subjet_size_fraction,
        subjet_pt_min_fraction=subjet_pt_min_fraction,
        subjet_dr_min=subjet_dr_min,
        trimmed_pt_min=trimmed_pt_min,
        trimmed_pt_max=trimmed_pt_max,
        trimmed_mass_min=trimmed_mass_min,
        trimmed_mass_max=trimmed_mass_max,
        shrink=shrink,
        shrink_mass=shrink_mass,
        compute_auxvars=compute_auxvars)

    if isinstance(inputs, HepMCInput):
        cluster_func = cluster_hepmc
    elif isinstance(inputs, h5.Dataset):
        cluster_func = cluster_hdf5
    else:
        cluster_func = cluster_iterable
        if not inspect.isgenerator(inputs):
            # handle case where input is just one event
            inputs = [inputs]
        kwargs['events'] = events

    if skip_failed:
        for event in cluster_func(inputs, **kwargs):
            if event is not None:
                yield event
    else:
        for event in cluster_func(inputs, **kwargs):
            yield event

# WIP
"""
def generate(gen_input, events,
             eta_max=5.,
             jet_size=1.0, subjet_size_fraction=0.5,
             subjet_pt_min_fraction=0.05,
             subjet_dr_min=0.,
             trimmed_pt_min=-1., trimmed_pt_max=-1.,
             trimmed_mass_min=-1., trimmed_mass_max=-1.,
             shrink=False, shrink_mass=-1,
             compute_auxvars=False,
             delphes=False,
             delphes_config='delphes_card_ATLAS_NoFastJet.tcl',
             delphes_random_state=0):

    if subjet_size_fraction <= 0 or subjet_size_fraction > 0.5:
        raise ValueError("subjet_size_fraction must be in the range (0, 0.5]")

    if not os.path.isfile(delphes_config):
        # use global config in share directory
        delphes_config = os.path.join(
            os.environ.get('DEEPJETS_SFT_DIR', '/usr/local'),
            'share/Delphes/cards', delphes_config)

    for event in _generate_events(
            gen_input, events,
            eta_max=eta_max,
            jet_size=jet_size,
            subjet_size_fraction=subjet_size_fraction,
            subjet_pt_min_fraction=subjet_pt_min_fraction,
            subjet_dr_min=subjet_dr_min,
            trimmed_pt_min=trimmed_pt_min,
            trimmed_pt_max=trimmed_pt_max,
            trimmed_mass_min=trimmed_mass_min,
            trimmed_mass_max=trimmed_mass_max,
            shrink=shrink,
            shrink_mass=shrink_mass,
            compute_auxvars=compute_auxvars,
            delphes=delphes,
            delphes_config=delphes_config,
            delphes_random_state=delphes_random_state):
        yield event
"""
