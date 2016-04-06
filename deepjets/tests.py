
from nose.tools import (raises, assert_raises, assert_true,
                        assert_equal, assert_almost_equal)

from deepjets.generate import generate_events, get_generator_input
from deepjets.clustering import cluster


def get_one_event(random_state=1, gen_params=None, **kwargs):
    gen_params = gen_params or dict()
    gen_input = get_generator_input('pythia', 'w.config',
                                    random_state=random_state,
                                    verbosity=0,
                                    **gen_params)
    particles = list(generate_events(gen_input, 1))[0]
    return list(cluster(particles, **kwargs))[0]


def test_generate_random_state():
    event = get_one_event()
    for i in range(3):
        # we should keep on getting the same result
        assert_equal(event.jets[0]['pT'], get_one_event().jets[0]['pT'])


def test_subjetiness():
    params_dict = {
        'PhaseSpace:pTHatMin': 250,
        'PhaseSpace:pTHatMax': 300}

    event_noshrink = get_one_event(
        gen_params=dict(params_dict=params_dict),
        compute_auxvars=True, shrink=False)
    event_shrink = get_one_event(
        gen_params=dict(params_dict=params_dict),
        compute_auxvars=True, shrink=True, shrink_mass=80)

    # shrinkage should only decrease pT
    assert_true(event_noshrink.jets[0]['pT'] >= event_shrink.jets[0]['pT'])

    # shrinkage will change values of nsubjetiness
    assert_true(event_noshrink.tau_1 != event_shrink.tau_1)
    assert_true(event_noshrink.tau_2 != event_shrink.tau_2)
    assert_true(event_noshrink.tau_3 != event_shrink.tau_3)
