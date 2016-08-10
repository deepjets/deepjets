from deepjets.generate import generate_events, get_generator_input
from deepjets.clustering import cluster
from deepjets.detector import reconstruct
from deepjets.preprocessing import preprocess, pixel_edges
from deepjets.utils import plot_jet_image
import numpy as np
from nose.tools import assert_true
from numpy.testing import assert_array_equal
import matplotlib.pyplot as plt


def generate_event(renorm_fac=1., factor_fac=1., random_state=1):
    # generate one event
    gen_input = get_generator_input(
        'pythia', 'w.config', random_state=random_state, verbosity=0,
        params_dict={
            'PhaseSpace:pTHatMin': 250,
            'PhaseSpace:pTHatMax': 300,
            'SigmaProcess:renormMultFac': renorm_fac,
            'SigmaProcess:factorMultFac': factor_fac})

    event = list(cluster(
        reconstruct(generate_events(gen_input, ignore_weights=True), random_state=1), events=1,
        jet_size=1.0, subjet_size=0.3,
        trimmed_pt_min=250, trimmed_pt_max=300,
        trimmed_mass_min=65, trimmed_mass_max=95))[0]

    edges = pixel_edges(
        jet_size=1.0,
        pixel_size=(0.1, 0.1),
        border_size=0)

    image = preprocess(
        event.subjets, event.trimmed_constit, edges,
        zoom=1.,
        normalize=True,
        out_width=25)

    return image


assert_true(generate_event().sum() > 0)
assert_array_equal(generate_event(), generate_event())

print "Renormalization scale uncertainty"
a = generate_event()
b = generate_event(renorm_fac=2.)
c = generate_event(renorm_fac=0.5)

print (b - a).sum() / a.sum()
print (c - a).sum() / a.sum()

print "Factorization scale uncertainty"
b = generate_event(factor_fac=2.)
c = generate_event(factor_fac=0.5)

print (b - a).sum() / a.sum()
print (c - a).sum() / a.sum()

print (generate_event(random_state=2) - a).sum() / a.sum()
