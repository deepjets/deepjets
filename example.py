from deepjets.generate import generate_events, get_generator_input
from deepjets.clustering import cluster
from deepjets.detector import reconstruct
from deepjets.preprocessing import preprocess, pixel_edges

gen_input = get_generator_input('pythia', 'w.config', random_state=1, verbosity=0)
edges = pixel_edges(
    jet_size=1.0,
    pixel_size=(0.1, 0.1),
    border_size=0)

for particles in generate_events(gen_input, events=1, ignore_weights=True):
    print particles  # numpy record array

for particle_jets in cluster(generate_events(gen_input, ignore_weights=True), events=1):
    print particle_jets  # jet struct
    image = preprocess(particle_jets.subjets, particle_jets.trimmed_constit, edges)

for towers in reconstruct(generate_events(gen_input, ignore_weights=True), events=1, random_state=1):
    print towers  # numpy record array

for tower_jets in cluster(reconstruct(generate_events(gen_input, ignore_weights=True), random_state=1), events=1):
    print tower_jets  # jet struct
    image = preprocess(tower_jets.subjets, tower_jets.trimmed_constit, edges)

