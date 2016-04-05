from deepjets.generate import get_generator_input, generate

gen_input = get_generator_input('pythia', 'w.config', random_state=1)

for event in generate(gen_input, 1):
    print event.jets
    print event.subjets
    print event.subjets.shape
    print event.trimmed_constit
    print event.trimmed_constit.shape
