from ._libdeepjets import run_delphes
from ._libdeepjets import HepMCInput
import h5py as h5
import inspect

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

    for event in run_delphes(particles, events, config, objects, random_state):
        yield event

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
