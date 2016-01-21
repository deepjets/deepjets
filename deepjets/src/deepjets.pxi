
cdef extern from "deepjets.h":
    cppclass Result:
        PseudoJet jet
        PseudoJet trimmed_jet
        vector[PseudoJet] subjets
        double shrinkage
        double subjet_dr
        double tau_1
        double tau_2
        double tau_3
        double trimmed_tau_1
        double trimmed_tau_2
        double trimmed_tau_3

    bool keep_event(Event&, int, double, double)
    void result_to_arrays(Result&, double*, double*, double*, double*)
    Result* get_jets(Event&, double, double, double, double, double, double, double, bool, double, bool)
