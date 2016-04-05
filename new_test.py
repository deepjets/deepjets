from deepjets.generate import get_generator_input
from deepjets._libdeepjets import generate_events

for event in generate_events(get_generator_input('pythia', 'w.config'), 1000):
    print event
