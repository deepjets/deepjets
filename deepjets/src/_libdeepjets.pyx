# cython: experimental_cpp_class_def=True, c_string_type=str, c_string_encoding=ascii

include "setup.pxi"
include "Pythia.pxi"
include "fastjet.pxi"
include "deepjets.pxi"

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

@cython.boundscheck(False)
@cython.wraparound(False)
def generate(string config, string xmldoc,
             int n_events,
             int random_seed=0,
             float beam_ecm=13000.,
             float eta_max=5.,
             float jet_size=0.6, float subjet_size=0.3,
             float jet_pt_min=12.5, float subjet_pt_min=0.05,
             float w_pt_min=-1, float w_pt_max=-1):
    """
    Generate and yield Pythia events
    """
    cdef int ievent;
    cdef Pythia* pythia = new Pythia(xmldoc, False)

    pythia.readString('Beams:eCM = {0}'.format(beam_ecm))
    pythia.readString('Random:setSeed = on')
    pythia.readString('Random:seed = {0}'.format(random_seed))
    pythia.readString("Init:showProcesses = on")
    pythia.readString("Init:showMultipartonInteractions = off")
    pythia.readString("Init:showChangedSettings = on")
    pythia.readString("Init:showChangedParticleData = off")
    pythia.readString("Next:numberShowInfo = 0")
    pythia.readString("Next:numberShowProcess = 0")
    pythia.readString("Next:numberShowEvent = 0")

    pythia.readFile(config)

    pythia.init()

    cdef int num_subjets = 0
    cdef int num_constit = 0

    cdef np.ndarray jet_arr = np.empty(3, dtype=np.double)
    cdef np.ndarray subjets_arr
    cdef np.ndarray constit_arr

    cdef Result* result

    try:
        ievent = 0
        while ievent < n_events:
            # Generate event. Quit if failure.
            if not pythia.next():
                raise RuntimeError("event generation aborted prematurely")
            if not keep_event(pythia.event, w_pt_min, w_pt_max):
                continue

            result = new Result()
            get_jets(pythia.event, result[0],
                     eta_max, jet_size, subjet_size,
                     jet_pt_min, subjet_pt_min)

            num_subjets = result.subjets.size()
            num_constit = 0
            for isubjet in range(result.subjets.size()):
                num_constit += result.subjets[isubjet].constituents().size()

            subjets_arr = np.empty((num_subjets, 3), dtype=np.double)
            constit_arr = np.empty((num_constit, 4), dtype=np.double)

            jets_to_arrays(result[0], <double*> jet_arr.data, <double*> subjets_arr.data,  <double*> constit_arr.data)
            del result

            yield jet_arr, subjets_arr, constit_arr
            ievent += 1
    finally:
        del pythia
