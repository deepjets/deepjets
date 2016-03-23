from ._libdeepjets import generate_events as _generate_events
from ._libdeepjets import PythiaInput, HepMCInput
import os


def get_generator_input(name, filename, **kwargs):
    """
    name may be 'pythia' or 'hepmc'
    filename may be the pythia config file or a HepMC file
    """
    name = name.lower().strip()
    if name == 'pythia':
        random_state = kwargs.pop('random_state', 0)
        beam_ecm = kwargs.pop('beam_ecm', 13000.)
        cut_on_pdgid = kwargs.pop('cut_on_pdgid', 0)
        pdgid_pt_min = kwargs.pop('pdgid_pt_min', -1)
        pdgid_pt_max = kwargs.pop('pdgid_pt_max', -1)
        xmldoc = os.path.join(
            os.environ.get('DEEPJETS_SFT_DIR', '/usr/local'),
            'share/Pythia8/xmldoc')
        gen_input = PythiaInput(filename, xmldoc, random_state, beam_ecm,
                                cut_on_pdgid, pdgid_pt_min, pdgid_pt_max,
                                kwargs)
    elif name == 'hepmc':
        gen_input = HepMCInput(filename)
        if kwargs:
            raise ValueError(
                "unrecognized parameters in kwargs: {0}".format(kwargs))
    else:
        raise ValueError(
            "no generator input available with name '{0}'".format(name))
    return gen_input


def generate(gen_input, nevents,
             eta_max=5.,
             jet_size=1.0, subjet_size_fraction=0.5,
             subjet_pt_min_fraction=0.05,
             subjet_dr_min=0.,
             trimmed_pt_min=10., trimmed_pt_max=-1.,
             shrink=False, shrink_mass=-1,
             compute_auxvars=False,
             delphes=False,
             delphes_config='delphes_card_ATLAS_NoFastJet.tcl',
             delphes_random_state=0):

    if not os.path.isfile(delphes_config):
        # use global config in share directory
        delphes_config = os.path.join(
            os.environ.get('DEEPJETS_SFT_DIR', '/usr/local'),
            'share/Delphes/cards', delphes_config)

    for event in _generate_events(
            gen_input, nevents,
            eta_max=eta_max,
            jet_size=jet_size,
            subjet_size_fraction=subjet_size_fraction,
            subjet_pt_min_fraction=subjet_pt_min_fraction,
            subjet_dr_min=subjet_dr_min,
            trimmed_pt_min=trimmed_pt_min,
            trimmed_pt_max=trimmed_pt_max,
            shrink=shrink,
            shrink_mass=shrink_mass,
            compute_auxvars=compute_auxvars,
            delphes=delphes,
            delphes_config=delphes_config,
            delphes_random_state=delphes_random_state):
        yield event
