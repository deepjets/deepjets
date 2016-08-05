from ._libdeepjets import generate_events as _generate_events
from ._libdeepjets import PythiaInput, HepMCInput
import os
from fnmatch import fnmatch

__all__ = [
    'generate_events',
    'PythiaInput', 'HepMCInput',
    'get_generator_input',
    ]


def get_generator_input(name, filename, **kwargs):
    """
    name may be 'pythia' or 'hepmc'
    filename may be the pythia config file or a HepMC file
    """
    name = name.lower().strip()
    if name == 'pythia':
        xmldoc = os.environ.get('PYTHIA8DATA', os.path.join(
            os.environ.get('DEEPJETS_SFT_DIR', '/usr/local'),
            'share/Pythia8/xmldoc'))
        gen_input = PythiaInput(filename, xmldoc, **kwargs)
    elif name == 'hepmc':
        gen_input = HepMCInput(filename)
        if kwargs:
            raise ValueError(
                "unrecognized parameters in kwargs: {0}".format(kwargs))
    else:
        raise ValueError(
            "no generator input available with name '{0}'".format(name))
    return gen_input


def generate_events(gen_input, events=-1, write_to='', ignore_weights=False, **kwargs):
    if isinstance(gen_input, basestring):
        if fnmatch(os.path.splitext(gen_input)[1], '.hepmc*'):
            gen_input = get_generator_input('hepmc', gen_input, **kwargs)
        else:
            gen_input = get_generator_input('pythia', gen_input, **kwargs)
    for event in _generate_events(gen_input, events, write_to, ignore_weights):
        yield event
