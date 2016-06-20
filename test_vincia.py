from deepjets.generate import generate_events

for event in generate_events('w_vincia.config', 1, write_to='vincia.hepmc', vincia=True):
    pass
