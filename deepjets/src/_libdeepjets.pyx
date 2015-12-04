# cython: experimental_cpp_class_def=True, c_string_type=str, c_string_encoding=ascii
import numpy as np
cimport numpy as np
np.import_array()

cimport cython

include "Pythia.pxi"


#@cython.boundscheck(False)
#@cython.wraparound(False)
def generate(string xmldoc, int n_events,
             int random_seed=0,
             float beam_ecm=13000.,
             float jet_size=0.6, float subjet_size=0.3,
             float jet_min_pt=12.5, float subjet_min_pt=0.05,
             float w_min_pt=-1, float w_max_pt=-1):
    """
    Generate and yield Pythia events
    """
    cdef int ievent;
    cdef Pythia* pythia = new Pythia(xmldoc, False)
    pythia.readString('Beams:eCM = {0}'.format(beam_ecm))
    pythia.readString('Random:setSeed = on')
    pythia.readString('Random:seed = {0}'.format(random_seed))

    # W' pair production.
    pythia.readString("NewGaugeBoson:ffbar2Wprime = on")
    pythia.readString("34:m0 = 700.0")

    # W' decays.
    pythia.readString("Wprime:coup2WZ = 1.0")
    pythia.readString("34:onMode = off")
    pythia.readString("34:onIfAny = 24")

    # W and Z decays.
    pythia.readString("24:onMode = off")
    pythia.readString("24:onIfAny = 1 2 3 4 5 6")
    pythia.readString("23:onMode = off")
    pythia.readString("23:onIfAny = 12 14 16")

    # Switch on/off particle data and event listings.
    pythia.readString("Init:showProcesses = off")
    pythia.readString("Init:showMultipartonInteractions = off")
    pythia.readString("Init:showChangedSettings = off")
    pythia.readString("Init:showChangedParticleData = off")
    pythia.readString("Next:numberShowInfo = 0")
    pythia.readString("Next:numberShowProcess = 0")
    pythia.readString("Next:numberShowEvent = 0")

    pythia.init()

    try:
        for ievent in range(n_events):
            # Generate event. Quit if failure.
            if not pythia.next():
                raise RuntimeError("event generation aborted prematurely")
            yield 1
    finally:
        del pythia
