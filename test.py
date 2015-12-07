from deepjets import generate

for leading_jet, subjets, constit in generate(100, w_pt_min=250, w_pt_max=300):
    print leading_jet, subjets.shape[0], constit.shape[0]
