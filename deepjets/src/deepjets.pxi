
cdef extern from "clustering.h":
    cppclass Result:
        PseudoJet jet
        PseudoJet trimmed_jet
        vector[PseudoJet] subjets
        vector[PseudoJet] constituents
        vector[PseudoJet] trimmed_constituents
        double shrinkage
        double subjet_dr
        double tau_1
        double tau_2
        double tau_3

    bool keep_pythia_event(Event&, int, double, double)
    void result_to_arrays(Result&, double*, double*, double*, double*)
    Result* get_jets(vector[PseudoJet]&, double, double, double,
                     double, double, double, double, double, bool,
                     double, bool)


cdef extern from "utils.h":
    IO_GenEvent* get_hepmc_reader(string)
    IO_GenEvent* get_hepmc_writer(string)
    void hepmc_to_pseudojet(GenEvent&, vector[PseudoJet]&, double)
    void hepmc_to_delphes(GenEvent* event, TDatabasePDG* pdg,
                          Delphes* delphes, TObjArray* all_particles,
                          TObjArray* stable_particles, TObjArray* partons) 
    void pythia_to_pseudojet(Event&, vector[PseudoJet]&, double)
    void pythia_to_delphes(Event&, Delphes*, TObjArray*, TObjArray*, TObjArray*)
    void delphes_to_pseudojet(TObjArray*, vector[PseudoJet]&)
    void delphes_to_array(TObjArray* input_array, double* array)
    GenEvent* pythia_to_hepmc(Pythia*)
    void hepmc_finalstate_particles(GenEvent*, vector[GenParticle*]&)
    void particles_to_array(vector[GenParticle*]& particles, double*)
    void array_to_delphes(int num_particles, double* particles, TDatabasePDG* pdg,
                          Delphes* delphes, TObjArray* all_particles,
                          TObjArray* stable_particles, TObjArray* partons) 
    void array_to_pseudojets(unsigned int size, unsigned int fields, double* array, vector[PseudoJet]& output, double eta_max)
