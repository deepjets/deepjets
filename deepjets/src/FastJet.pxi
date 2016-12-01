
cdef extern from "fastjet/PseudoJet.hh" namespace "fastjet":
    cdef cppclass PseudoJet:
        vector[PseudoJet] constituents()
