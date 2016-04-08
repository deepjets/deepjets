# cython: experimental_cpp_class_def=True, c_string_type=str, c_string_encoding=ascii

include "setup.pxi"
include "Pythia.pxi"
include "HepMC.pxi"
include "FastJet.pxi"
include "Delphes.pxi"
include "deepjets.pxi"

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# for jet clustering:
dtype_jet = np.dtype([('pT', DTYPE), ('eta', DTYPE), ('phi', DTYPE), ('mass', DTYPE)])
dtype_constit = np.dtype([('ET', DTYPE), ('eta', DTYPE), ('phi', DTYPE)])
# for pythia/hepmc:
dtype_particle = np.dtype([('E', DTYPE), ('px', DTYPE), ('py', DTYPE), ('pz', DTYPE), ('mass', DTYPE),
                           ('prodx', DTYPE), ('prody', DTYPE), ('prodz', DTYPE), ('prodt', DTYPE),
                           ('pdgid', DTYPE)])
# for Delphes output
dtype_fourvect = np.dtype([('E', DTYPE), ('px', DTYPE), ('py', DTYPE), ('pz', DTYPE)])

include "generate.pyx"
include "clustering.pyx"
include "detector.pyx"
