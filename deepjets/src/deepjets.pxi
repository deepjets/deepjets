
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

    bool keep_pythia_event(Event&, int, double, double)
    void result_to_arrays(Result&, double*, double*, double*, double*)
    Result* get_jets(vector[PseudoJet]&, double, double, double, double, double, double, bool, double, bool)
    
    IO_GenEvent* get_hepmc_reader(string)
    void hepmc_to_pseudojet(GenEvent&, vector[PseudoJet]&, double)
    void pythia_to_pseudojet(Event&, vector[PseudoJet]&, double)
