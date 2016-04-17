from ._libdeepjets import (DelphesWrapper, reconstruct_mc, reconstruct_numpy,
                           reconstruct_hdf5, reconstruct_iterable)
from ._libdeepjets import MCInput
import numpy as np
import h5py as h5
import inspect
import os

__all__ = [
    'reconstruct',
    ]


def reconstruct(particles, events=-1,
                config='delphes_card_ATLAS_NoFastJet.tcl',
                objects='Calorimeter/towers',
                random_state=0):

    if not os.path.isfile(config):
        # use global config in share directory
        config = os.path.join(
            os.environ.get('DEEPJETS_SFT_DIR', '/usr/local'),
            'share/Delphes/cards', config)
    delphes = DelphesWrapper(config, random_state, objects)

    kwargs = dict()

    if isinstance(particles, MCInput):
        reco_func = reconstruct_mc
    elif isinstance(particles, h5.Dataset):
        reco_func = reconstruct_hdf5
    else:
        reco_func = reconstruct_iterable
        kwargs['events'] = events

        if not inspect.isgenerator(particles) and not isinstance(particles, list):
            # handle case where input is just one event
            particles = [particles]

    for event in reco_func(delphes, particles, **kwargs):
        yield event
