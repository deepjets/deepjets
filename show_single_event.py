from deepjets.tests import get_one_event

def print_event(event):
    print event.jets
    print event.subjets
    print event.subjets.shape
    print event.trimmed_constit
    print event.trimmed_constit.shape


params_dict = {
    'PhaseSpace:pTHatMin': 250,
    'PhaseSpace:pTHatMax': 300}

gen_params = dict(
    verbosity=0,
    params_dict=params_dict)

event = get_one_event(random_state=10, shrink=False, gen_params=gen_params)
print_event(event)
event = get_one_event(random_state=10, shrink=True, gen_params=gen_params)
print_event(event)
