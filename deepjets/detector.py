from ._libdeepjets import (DelphesWrapper, reconstruct_mc, reconstruct_numpy,
                           reconstruct_hdf5, reconstruct_iterable)
from ._libdeepjets import MCInput
import numpy as np
import h5py as h5
import inspect
import os
import logging

log = logging.getLogger(__name__)

__all__ = [
    'reconstruct',
    ]


def reconstruct(particles, events=-1,
                config='delphes_card_ATLAS_NoFastJet.tcl',
                objects='Calorimeter/towers',
                random_state=0):

    if not os.path.exists(config):
        internal_config = os.path.join(
            os.environ.get('DEEPJETS_DIR'),
            'config', 'delphes', config)
        if not os.path.isabs(config) and os.path.exists(internal_config):
            log.warning("{0} does not exist but using internal "
                        "config with the same name instead: {1}".format(
                            config, internal_config))
            config = internal_config
        else:
            raise IOError("Delphes config not found: {0}".format(config))
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
