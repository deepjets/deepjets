from deepjets import generate

for leading_jet, subjets, constit in generate('wprime.config', 10,
                                             w_pt_min=250, w_pt_max=300):
    print leading_jet, subjets.shape[0], constit.shape[0]

for leading_jet, subjets, constit in generate('qcd.config', 10):
    print leading_jet, subjets.shape[0], constit.shape[0]
