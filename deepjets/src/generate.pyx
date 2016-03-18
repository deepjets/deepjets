import tempfile

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
def generate_pythia(string config, string xmldoc,
                    int n_events,
                    int random_state=0,
                    float beam_ecm=13000.,
                    float eta_max=5.,
                    float jet_size=0.6,
                    float subjet_size_fraction=0.5,
                    float subjet_pt_min_fraction=0.05,
                    float subjet_dr_min=0.,
                    float trimmed_pt_min=10., float trimmed_pt_max=-1., 
                    bool shrink=False, float shrink_mass=-1.,
                    int cut_on_pdgid=0, float pdgid_pt_min=-1, float pdgid_pt_max=-1,
                    params_dict=None,
                    bool compute_auxvars=False,
                    bool delphes=False,
                    string delphes_config=''):
    """
    Generate Pythia events and yield jet and constituent arrays
    """
    if subjet_size_fraction <= 0 or subjet_size_fraction > 0.5:
        raise ValueError("subjet_size_fraction must be in the range (0, 0.5]")
    
    # Delphes init
    cdef ExRootConfReader* delphes_conf_reader = NULL
    cdef Delphes* modular_delphes = NULL
    cdef TObjArray* delphes_all_particles = NULL
    cdef TObjArray* delphes_stable_particles = NULL
    cdef TObjArray* delphes_partons = NULL
    cdef TObjArray* delphes_input_array = NULL

    if delphes:
        delphes_config_reader = new ExRootConfReader()
        delphes_config_reader.ReadFile(delphes_config.c_str())
        # Set Delhes' random seed. Only possible through a config file...
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write("set RandomSeed {0:d}\n".format(random_state))
            tmp.flush()
            delphes_config_reader.ReadFile(tmp.name)
        modular_delphes = new Delphes("Delphes")
        modular_delphes.SetConfReader(delphes_config_reader)
        delphes_all_particles = modular_delphes.ExportArray("allParticles")
        delphes_stable_particles = modular_delphes.ExportArray("stableParticles")
        delphes_partons = modular_delphes.ExportArray("partons")
        modular_delphes.InitTask()
        delphes_input_array = modular_delphes.ImportArray("Calorimeter/towers")
    
    # Pythia init
    cdef int ievent;
    cdef Pythia* pythia = new Pythia(xmldoc, False)

    pythia.readString("Init:showProcesses = on")
    pythia.readString("Init:showMultipartonInteractions = off")
    pythia.readString("Init:showChangedSettings = on")
    pythia.readString("Init:showChangedParticleData = off")
    pythia.readString("Next:numberShowInfo = 0")
    pythia.readString("Next:numberShowProcess = 0")
    pythia.readString("Next:numberShowEvent = 0")
    pythia.readFile(config)  # read user config after options above
    pythia.readString('Beams:eCM = {0}'.format(beam_ecm))
    pythia.readString('Random:setSeed = on')
    pythia.readString('Random:seed = {0}'.format(random_state))
    if params_dict is not None:
        for param, value in params_dict.items():
            pythia.readString('{0} = {1}'.format(param, value))
    pythia.init()

    cdef int num_jet_constit = 0
    cdef int num_subjets = 0
    cdef int num_subjets_constit = 0

    cdef np.ndarray jet_arr  # contains jet and trimmed jet
    cdef np.ndarray subjet_arr  # contains subjets (that sum to trimmed jet)
    cdef np.ndarray jet_constit_arr  # contains original jet constituents
    cdef np.ndarray subjet_constit_arr  # contains trimmed jet constituents

    cdef Result* result
    cdef vector[PseudoJet] particles

    dtype_jet = np.dtype([('pT', DTYPE), ('eta', DTYPE), ('phi', DTYPE), ('mass', DTYPE)])
    dtype_constit = np.dtype([('ET', DTYPE), ('eta', DTYPE), ('phi', DTYPE)])

    try:
        ievent = 0
        while ievent < n_events:
            # generate event and quit if failure
            if not pythia.next():
                raise RuntimeError("event generation aborted prematurely")
            
            if not keep_pythia_event(pythia.event, cut_on_pdgid, pdgid_pt_min, pdgid_pt_max):
                # event doesn't pass our truth-level cuts
                continue

            particles.clear()

            if delphes:
                modular_delphes.Clear()
                # convert Pythia particles into Delphes candidates
                pythia_to_delphes(pythia.event, modular_delphes,
                                  delphes_all_particles,
                                  delphes_stable_particles,
                                  delphes_partons)
                # run Delphes reconstruction
                modular_delphes.ProcessTask()
                # convert Delphes reconstructed eflow candidates into pseudojets
                delphes_to_pseudojet(delphes_input_array, particles)
            else:
                # convert Pythia particles directly into pseudojets
                pythia_to_pseudojet(pythia.event, particles, eta_max)
            
            # run jet clustering
            result = get_jets(particles,
                              jet_size, subjet_size_fraction,
                              subjet_pt_min_fraction,
                              subjet_dr_min,
                              trimmed_pt_min, trimmed_pt_max,
                              shrink, shrink_mass,
                              compute_auxvars)

            if result == NULL:
                # didn't find any jets passing cuts in this event
                continue

            num_jet_constit = result.jet.constituents().size()
            num_subjets = result.subjets.size()
            num_subjets_constit = 0
            for isubjet in range(result.subjets.size()):
                num_subjets_constit += result.subjets[isubjet].constituents().size()
            
            jet_arr = np.empty((2,), dtype=dtype_jet)
            subjet_arr = np.empty((num_subjets,), dtype=dtype_jet)
            jet_constit_arr = np.empty((num_jet_constit,), dtype=dtype_constit)
            subjet_constit_arr = np.empty((num_subjets_constit,), dtype=dtype_constit)

            result_to_arrays(result[0],
                             <DTYPE_t*> jet_arr.data,
                             <DTYPE_t*> subjet_arr.data,
                             <DTYPE_t*> jet_constit_arr.data,
                             <DTYPE_t*> subjet_constit_arr.data)
            
            if compute_auxvars:
                auxdict = {
                    'subjet_dr': result.subjet_dr,
                    'tau_1': result.tau_1,
                    'tau_2': result.tau_2,
                    'tau_3': result.tau_3,
                    }
                yield jet_arr, subjet_arr, jet_constit_arr, subjet_constit_arr, result.shrinkage, auxdict
            else:
                yield jet_arr, subjet_arr, jet_constit_arr, subjet_constit_arr, result.shrinkage

            del result
            ievent += 1
        pythia.stat()
    finally:
        del pythia
        del modular_delphes
        del delphes_config_reader


@cython.boundscheck(False)
@cython.wraparound(False)
def read_hepmc(string filename,
               int n_events,
               float eta_max=5.,
               float jet_size=0.6,
               float subjet_size_fraction=0.5,
               float subjet_pt_min_fraction=0.05,
               float subjet_dr_min=0.,
               float trimmed_pt_min=10., float trimmed_pt_max=-1., 
               bool shrink=False, float shrink_mass=-1.,
               bool compute_auxvars=False):
    """
    Read HepMC files and yield jet and constituent arrays
    """
    if subjet_size_fraction <= 0 or subjet_size_fraction > 0.5:
        raise ValueError("subjet_size_fraction must be in the range (0, 0.5]")

    cdef int ievent;

    cdef int num_jet_constit = 0
    cdef int num_subjets = 0
    cdef int num_subjets_constit = 0

    cdef np.ndarray jet_arr  # contains jet and trimmed jet
    cdef np.ndarray subjet_arr  # contains subjets (that sum to trimmed jet)
    cdef np.ndarray jet_constit_arr  # contains original jet constituents
    cdef np.ndarray subjet_constit_arr  # contains trimmed jet constituents

    cdef Result* result

    cdef IO_GenEvent* hepmc_reader
    cdef GenEvent* event
    cdef vector[PseudoJet] particles

    hepmc_reader = get_hepmc_reader(filename)

    dtype_jet = np.dtype([('pT', DTYPE), ('eta', DTYPE), ('phi', DTYPE), ('mass', DTYPE)])
    dtype_constit = np.dtype([('ET', DTYPE), ('eta', DTYPE), ('phi', DTYPE)])

    try:
        ievent = 0
        while n_events < 0 or ievent < n_events:
            # get next event
            event = hepmc_reader.read_next_event()
            if event == NULL:
                break

            particles.clear()
            hepmc_to_pseudojet(event[0], particles, eta_max)
            del event

            result = get_jets(particles,
                              jet_size, subjet_size_fraction,
                              subjet_pt_min_fraction,
                              subjet_dr_min,
                              trimmed_pt_min, trimmed_pt_max,
                              shrink, shrink_mass,
                              compute_auxvars)

            if result == NULL:
                # didn't find any jets passing cuts in this event
                continue

            num_jet_constit = result.jet.constituents().size()
            num_subjets = result.subjets.size()
            num_subjets_constit = 0
            for isubjet in range(result.subjets.size()):
                num_subjets_constit += result.subjets[isubjet].constituents().size()
            
            jet_arr = np.empty((2,), dtype=dtype_jet)
            subjet_arr = np.empty((num_subjets,), dtype=dtype_jet)
            jet_constit_arr = np.empty((num_jet_constit,), dtype=dtype_constit)
            subjet_constit_arr = np.empty((num_subjets_constit,), dtype=dtype_constit)

            result_to_arrays(result[0],
                             <DTYPE_t*> jet_arr.data,
                             <DTYPE_t*> subjet_arr.data,
                             <DTYPE_t*> jet_constit_arr.data,
                             <DTYPE_t*> subjet_constit_arr.data)
            
            if compute_auxvars:
                auxdict = {
                    'subjet_dr': result.subjet_dr,
                    'tau_1': result.tau_1,
                    'tau_2': result.tau_2,
                    'tau_3': result.tau_3,
                    }
                yield jet_arr, subjet_arr, jet_constit_arr, subjet_constit_arr, result.shrinkage, auxdict
            else:
                yield jet_arr, subjet_arr, jet_constit_arr, subjet_constit_arr, result.shrinkage

            del result
            ievent += 1
    finally:
        del hepmc_reader
