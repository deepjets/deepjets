
from nose.tools import (raises, assert_raises, assert_true,
                        assert_equal, assert_almost_equal)

from deepjets.generate import generate, get_generator_input


def get_one_event(random_state=1, gen_params=None, **kwargs):
    if gen_params is None:
        gen_params = dict()
    gen_input = get_generator_input('pythia', 'w.config',
                                    random_state=random_state,
                                    **gen_params)
    return list(generate(gen_input, events=1, **kwargs))[0]


def test_generate_random_state():
    event = get_one_event()
    for i in range(3):
        # we should keep on getting the same result
        assert_equal(event.jets[0]['pT'], get_one_event().jets[0]['pT'])


def test_subjetiness():
    params_dict = {
        'PhaseSpace:pTHatMin': 250,
        'PhaseSpace:pTHatMax': 300}

    event_noshrink = get_one_event(gen_params=dict(params_dict=params_dict),
                                   compute_auxvars=True, shrink=False)
    event_shrink = get_one_event(gen_params=dict(params_dict=params_dict),
                                 compute_auxvars=True, shrink=True, shrink_mass=80)

    # original jet should be the same
    # shrinkage only affects the subjets
    assert_equal(event_noshrink.jets[0]['pT'], event_shrink.jets[0]['pT'])

    # shrinkage does not affect nsubjetiness since it is calculated on the
    # original clustering with fixed size
    assert_equal(event_noshrink.tau_1, event_shrink.tau_1)
    assert_equal(event_noshrink.tau_2, event_shrink.tau_2)
    assert_equal(event_noshrink.tau_3, event_shrink.tau_3)
