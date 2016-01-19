
cdef extern from "deepjets.h":
    cppclass Result:
        PseudoJet jet
        vector[PseudoJet] subjets
        double shrinkage

    bool keep_event(Event&, int, double, double)
    void result_to_arrays(Result&, double*, double*, double*, double*)
    Result* get_jets(Event&, double, double, double, double, double, bool, double)
