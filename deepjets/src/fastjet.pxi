
cdef extern from "Pythia8Plugins/FastJet3.h" namespace "fastjet":
    cdef cppclass PseudoJet:
        vector[PseudoJet] constituents()
