
cdef extern from "deepjets.h":
    cppclass Result:
        PseudoJet jet
        vector[PseudoJet] subjets

    bool keep_event(Event&, int, double, double)
    void jets_to_arrays(Result&, double*, double*, double*)
    Result* get_jets(Event&, double, double, double, double, double, bool)
