from ._libdeepjets import generate as _generate
import os

def generate(nevents):
    xmldoc = os.path.join(os.environ['PYTHIADIR'], 'share/Pythia8/xmldoc')
    for event in _generate(xmldoc, nevents):
        yield event
