from ._libdeepjets import generate_pythia as _generate
import os

def generate(config, nevents,
             random_seed=0,
             beam_ecm=13000.,
             eta_max=5.,
             jet_size=1.2, subjet_size_fraction=0.5,
             jet_pt_min=12.5, subjet_pt_min_fraction=0.05,
             shrink=False,
             cut_on_pdgid=0, pt_min=-1, pt_max=-1,
             params_dict=None):
    xmldoc = os.path.join(os.environ.get('DEEPJETS_SFT_DIR', '/usr/local'),
                          'share/Pythia8/xmldoc')
    for event in _generate(config, xmldoc, nevents,
                           random_seed=random_seed,
                           beam_ecm=beam_ecm,
                           eta_max=eta_max,
                           jet_size=jet_size,
                           subjet_size_fraction=subjet_size_fraction,
                           jet_pt_min=jet_pt_min,
                           subjet_pt_min_fraction=subjet_pt_min_fraction,
                           shrink=shrink,
                           cut_on_pdgid=cut_on_pdgid,
                           pt_min=pt_min, pt_max=pt_max,
                           params_dict=params_dict):
        yield event
