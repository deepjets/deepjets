
cdef extern from "deepjets.h":
    cppclass Result:
        PseudoJet jet
        vector[PseudoJet] subjets

    bool keep_event(Event&, double, double)
    void jets_to_arrays(Result&, double*, double*, double*)
    void get_jets(Event&, Result&, double, double, double, double, double)
