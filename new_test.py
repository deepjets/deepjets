from deepjets.generate import get_generator_input
from deepjets._libdeepjets import generate_events, cluster_numpy

for event in generate_events(get_generator_input('pythia', 'w.config'), 1000):
    jets = cluster_numpy(event, 5., 1.0, 0.5, 0.03, 0.03, -1, -1, -1, -1, True, 81, True)
    if jets:
        print jets.subjets
