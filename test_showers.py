from deepjets.generate import generate_events

for event in generate_events('w_vincia.config', 1, write_to='vincia.hepmc', shower='vincia', random_state=1, verbosity=0):
    pass

for event, weight in generate_events('w.config', 1, write_to='dire.hepmc', shower='dire', random_state=1, verbosity=0):
    print weight
