import os
from pkg_resources import resource_filename


__all__ = [
    'get_filepath',
]


def get_filepath(name='sherpa_wz.hepmc'):
    return resource_filename('deepjets', os.path.join('testdata', name))
