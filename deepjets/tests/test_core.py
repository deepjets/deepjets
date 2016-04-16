
from nose.tools import (raises, assert_raises, assert_true,
                        assert_equal, assert_not_equal, assert_almost_equal)

from deepjets.generate import generate_events, get_generator_input
from deepjets.clustering import cluster
from deepjets.detector import reconstruct, DelphesWrapper
from deepjets.testdata import get_filepath
from deepjets.samples import create_event_datasets
from deepjets.generate import HepMCInput, PythiaInput


import tempfile
import h5py as h5
import numpy as np


def get_one_event(random_state=1, gen_params=None, **kwargs):
    gen_params = gen_params or dict()
    gen_input = get_generator_input('pythia', 'w.config',
                                    random_state=random_state,
                                    verbosity=0,
                                    **gen_params)
    return list(cluster(generate_events(gen_input), 1, **kwargs))[0]


def get_one_event_reco(pythia_random_state=1, delphes_random_state=1,
                       gen_params=None, **kwargs):
    gen_params = gen_params or dict()
    gen_input = get_generator_input('pythia', 'w.config',
                                    random_state=pythia_random_state,
                                    verbosity=0,
                                    **gen_params)
    return list(cluster(reconstruct(generate_events(gen_input),
                                    random_state=delphes_random_state),
                                    1, **kwargs))[0]


def test_cluster_generate_length():
    assert_equal(len(list(cluster(generate_events('w.config', verbosity=0), 1))), 1)
    assert_equal(len(list(cluster(generate_events('w.config', verbosity=0), 10))), 10)
    assert_equal(len(list(cluster(generate_events('w.config', verbosity=0), 100))), 100)


def test_reconstruct_generate_length():
    assert_equal(len(list(reconstruct(generate_events('w.config', verbosity=0), 1))), 1)
    assert_equal(len(list(reconstruct(generate_events('w.config', verbosity=0), 10))), 10)
    assert_equal(len(list(reconstruct(generate_events('w.config', verbosity=0), 100))), 100)


def test_cluster_reconstruct_generate_length():
    assert_equal(len(list(cluster(reconstruct(generate_events('w.config', verbosity=0)), 1))), 1)
    assert_equal(len(list(cluster(reconstruct(generate_events('w.config', verbosity=0)), 10))), 10)
    assert_equal(len(list(cluster(reconstruct(generate_events('w.config', verbosity=0)), 100))), 100)


def test_generate_random_state():
    event = get_one_event(random_state=1)
    for i in range(3):
        # we should keep on getting the same result
        assert_equal(event.jets[0]['pT'], get_one_event(random_state=1).jets[0]['pT'])
    for i in range(3):
        # we should get different results
        assert_not_equal(event.jets[0]['pT'], get_one_event(random_state=0).jets[0]['pT'])


def test_delphes_random_state():
    event = get_one_event_reco(delphes_random_state=1)
    for i in range(3):
        # we should keep on getting the same result
        assert_equal(event.jets[0]['pT'], get_one_event_reco(delphes_random_state=1).jets[0]['pT'])
    for i in range(3):
        # we should get different results
        assert_not_equal(event.jets[0]['pT'], get_one_event_reco(delphes_random_state=0).jets[0]['pT'])


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
    assert_not_equal(event_noshrink.tau_1, event_shrink.tau_1)
    assert_not_equal(event_noshrink.tau_2, event_shrink.tau_2)
    assert_not_equal(event_noshrink.tau_3, event_shrink.tau_3)


def test_hdf5_vs_direct_hepmc():
    testfile = get_filepath('sherpa_wz.hepmc')

    with tempfile.NamedTemporaryFile() as tmp:
        with h5.File(tmp.name, 'w') as h5output:
            create_event_datasets(h5output, 1)
            h5output['events'][0] = list(generate_events(HepMCInput(testfile), 1))[0]
        h5input = h5.File(tmp.name, 'r')
        jets = list(cluster(reconstruct(h5input['events'], random_state=1)))[0]
    jets_direct = list(cluster(reconstruct(HepMCInput(testfile), random_state=1)))[0]

    assert_equal(jets.jets[0]['pT'], jets_direct.jets[0]['pT'])


def test_hdf5_vs_direct_pythia():
    testfile = get_filepath('pythia_wz.config')

    with tempfile.NamedTemporaryFile() as tmp:
        with h5.File(tmp.name, 'w') as h5output:
            create_event_datasets(h5output, 1)
            h5output['events'][0] = list(
                generate_events(
                    get_generator_input('pythia', testfile, verbosity=0,
                                        random_state=1), 1))[0]
        h5input = h5.File(tmp.name, 'r')
        jets = list(cluster(reconstruct(h5input['events'], random_state=1)))[0]
    jets_direct = list(
        cluster(
            reconstruct(
                generate_events(
                    get_generator_input('pythia', testfile, verbosity=0,
                                        random_state=1), 1), random_state=1)))[0]

    assert_true(jets.jets[0]['pT'] > 0)
    assert_equal(jets.jets[0]['pT'], jets_direct.jets[0]['pT'])
