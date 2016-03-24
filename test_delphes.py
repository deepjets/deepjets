from deepjets.generate import get_generator_input, generate

gen_input = get_generator_input('pythia', 'w.config')

for pythia, delphes in generate(gen_input, 10, delphes=True):
    print pythia.jets
    if delphes is not None:
        print delphes.jets
    print '========'
