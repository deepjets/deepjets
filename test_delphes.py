from deepjets.generate import generate
for thing in generate('qcd.config', 100000, delphes=True, random_state=4):
    pass
