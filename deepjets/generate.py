from ._libdeepjets import generate_pythia as _generate
import os

def generate(config, nevents,
             random_state=None,
             beam_ecm=13000.,
             eta_max=5.,
             jet_size=1.0, subjet_size_fraction=0.5,
             subjet_pt_min_fraction=0.05,
             subjet_dr_min=0.,
             trimmed_pt_min=10., trimmed_pt_max=-1.,
             shrink=False, shrink_mass=-1,
             cut_on_pdgid=0, pdgid_pt_min=-1, pdgid_pt_max=-1,
             params_dict=None,
             compute_auxvars=False):
    if random_state is None:
        # Pythia will use time
        random_state = 0
    xmldoc = os.path.join(os.environ.get('DEEPJETS_SFT_DIR', '/usr/local'),
                          'share/Pythia8/xmldoc')
    for event in _generate(config, xmldoc, nevents,
                           random_state=random_state,
                           beam_ecm=beam_ecm,
                           eta_max=eta_max,
                           jet_size=jet_size,
                           subjet_size_fraction=subjet_size_fraction,
                           subjet_pt_min_fraction=subjet_pt_min_fraction,
                           subjet_dr_min=subjet_dr_min,
                           trimmed_pt_min=trimmed_pt_min,
                           trimmed_pt_max=trimmed_pt_max,
                           shrink=shrink,
                           shrink_mass=shrink_mass,
                           cut_on_pdgid=cut_on_pdgid,
                           pdgid_pt_min=pdgid_pt_min,
                           pdgid_pt_max=pdgid_pt_max,
                           params_dict=params_dict,
                           compute_auxvars=compute_auxvars):
        yield event
