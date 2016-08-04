import dask.array as da
from dask.diagnostics import ProgressBar
from deepjets.samples import make_flat_images
ret = make_flat_images('/coepp/cephfs/mel/edawe/deepjets/events/pythia/images/default2/qcd_j1p0_sj0p30_delphes_jets_pileup_images.h5', 250, 300, mass_min=50, mass_max=110)

images, auxvars, weights = ret

print images
print weights

print auxvars
print auxvars['generator_weights']

print "average"
with ProgressBar():
    avg_image = da.tensordot(images, weights, axes=(0, 0)).compute() / weights.sum()
print avg_image
