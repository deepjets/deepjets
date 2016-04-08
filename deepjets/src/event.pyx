import tempfile

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
dtype_jet = np.dtype([('pT', DTYPE), ('eta', DTYPE), ('phi', DTYPE), ('mass', DTYPE)])
dtype_constit = np.dtype([('ET', DTYPE), ('eta', DTYPE), ('phi', DTYPE)])
dtype_particle = np.dtype([('E', DTYPE), ('px', DTYPE), ('py', DTYPE), ('pz', DTYPE), ('mass', DTYPE),
                           ('prodx', DTYPE), ('prody', DTYPE), ('prodz', DTYPE), ('prodt', DTYPE),
                           ('pdgid', DTYPE)])


cdef class Jets:
    cdef readonly np.ndarray jets  # contains jet and trimmed jet
    cdef readonly np.ndarray subjets  # contains subjets (that sum to trimmed jet)
    cdef readonly np.ndarray constit  # contains original jet constituents
    cdef readonly np.ndarray trimmed_constit  # contains trimmed jet constituents
    cdef readonly float shrinkage
    cdef readonly float subjet_dr
    cdef readonly float tau_1
    cdef readonly float tau_2
    cdef readonly float tau_3


cdef void jets_from_result(Jets jets, Result* result):
    cdef int num_jet_constit
    cdef int num_subjets
    cdef int num_trimmed_constit

    num_jet_constit = result.constituents.size()
    num_subjets = result.subjets.size()
    num_trimmed_constit = result.trimmed_constituents.size()

    jets.jets = np.empty((2,), dtype=dtype_jet)
    jets.subjets = np.empty((num_subjets,), dtype=dtype_jet)
    jets.constit = np.empty((num_jet_constit,), dtype=dtype_constit)
    jets.trimmed_constit = np.empty((num_trimmed_constit,), dtype=dtype_constit)

    result_to_arrays(result[0],
                     <DTYPE_t*> jets.jets.data,
                     <DTYPE_t*> jets.subjets.data,
                     <DTYPE_t*> jets.constit.data,
                     <DTYPE_t*> jets.trimmed_constit.data)
    jets.shrinkage = result.shrinkage 
    jets.subjet_dr = result.subjet_dr
    jets.tau_1 = result.tau_1
    jets.tau_2 = result.tau_2
    jets.tau_3 = result.tau_3


@cython.boundscheck(False)
@cython.wraparound(False)
def generate_events(GeneratorInput gen_input, int n_events):
    """
    Generate events (or read HepMC) and yield numpy arrays of particles
    """
    cdef np.ndarray particle_array
    cdef GenEvent* event
    cdef vector[GenParticle*] particles
    cdef int ievent = 0;
    if n_events < 0:
        ievent = n_events - 1
    while ievent < n_events:
        if not gen_input.get_next_event():
            continue
        event = gen_input.get_hepmc()
        hepmc_finalstate_particles(event, particles)
        particle_array = np.empty((particles.size(),), dtype=dtype_particle)
        particles_to_array(particles, <DTYPE_t*> particle_array.data)
        yield particle_array
        del event
        if n_events > 0:
            ievent += 1
    gen_input.finish()


cdef object cluster_pseudojets(
        vector[PseudoJet] pseudojets,
        float jet_size,
        float subjet_size_fraction,
        float subjet_pt_min_fraction,
        float subjet_dr_min,
        float trimmed_pt_min, float trimmed_pt_max, 
        float trimmed_mass_min, float trimmed_mass_max,
        bool shrink, float shrink_mass,
        bool compute_auxvars):
    """
    Perform jet clustering on a single event with FastJet as defined in
    clustering.h in the get_jets function.
    Return a Jets struct defined above.
    """
    cdef Result* result
    # run jet clustering
    result = get_jets(pseudojets,
                      jet_size, subjet_size_fraction,
                      subjet_pt_min_fraction,
                      subjet_dr_min,
                      trimmed_pt_min, trimmed_pt_max,
                      trimmed_mass_min, trimmed_mass_max,
                      shrink, shrink_mass,
                      compute_auxvars)
    if result == NULL:
        # didn't find any jets passing cuts in this event
        return None
    jets = Jets()
    jets_from_result(jets, result)
    del result
    return jets


@cython.boundscheck(False)
@cython.wraparound(False)
def cluster_numpy(np.ndarray particles,
                  float eta_max,
                  float jet_size,
                  float subjet_size_fraction,
                  float subjet_pt_min_fraction,
                  float subjet_dr_min,
                  float trimmed_pt_min, float trimmed_pt_max, 
                  float trimmed_mass_min, float trimmed_mass_max,
                  bool shrink, float shrink_mass,
                  bool compute_auxvars):
    """
    Perform jet clustering on a numpy array of particles.
    See cluster_pseudojets above.
    """
    cdef vector[PseudoJet] pseudojets
    # convert numpy array into vector of pseudojets
    array_to_pseudojets(particles.shape[0], <DTYPE_t*> particles.data,
                        pseudojets, eta_max)
    return cluster_pseudojets(
        pseudojets,
        jet_size,
        subjet_size_fraction,
        subjet_pt_min_fraction,
        subjet_dr_min,
        trimmed_pt_min, trimmed_pt_max,
        trimmed_mass_min, trimmed_mass_max,
        shrink, shrink_mass,
        compute_auxvars)

    
@cython.boundscheck(False)
@cython.wraparound(False)
def cluster_hdf5(dataset,
                 float eta_max,
                 float jet_size,
                 float subjet_size_fraction,
                 float subjet_pt_min_fraction,
                 float subjet_dr_min,
                 float trimmed_pt_min, float trimmed_pt_max, 
                 float trimmed_mass_min, float trimmed_mass_max,
                 bool shrink, float shrink_mass,
                 bool compute_auxvars):
    """
    Perform jet clustering on an array of particle arrays.
    Yield the clustering for each event.
    See cluster_pseudojets above.
    """
    cdef vector[PseudoJet] pseudojets
    cdef int num_events = len(dataset)
    cdef np.ndarray particles
    cdef int ievent
    for ievent in range(num_events):
        particles = dataset[ievent]
        # convert numpy array into vector of pseudojets
        array_to_pseudojets(particles.shape[0], <DTYPE_t*> particles.data,
                            pseudojets, eta_max)
        yield cluster_pseudojets(
            pseudojets,
            jet_size,
            subjet_size_fraction,
            subjet_pt_min_fraction,
            subjet_dr_min,
            trimmed_pt_min, trimmed_pt_max,
            trimmed_mass_min, trimmed_mass_max,
            shrink, shrink_mass,
            compute_auxvars)

        
@cython.boundscheck(False)
@cython.wraparound(False)
def cluster_iterable(iterable,
                     int events,
                     bool skip_failed,
                     float eta_max,
                     float jet_size,
                     float subjet_size_fraction,
                     float subjet_pt_min_fraction,
                     float subjet_dr_min,
                     float trimmed_pt_min, float trimmed_pt_max, 
                     float trimmed_mass_min, float trimmed_mass_max,
                     bool shrink, float shrink_mass,
                     bool compute_auxvars):
    """
    Perform jet clustering on an array of particle arrays.
    Yield the clustering for each event.
    See cluster_pseudojets above.
    """
    cdef vector[PseudoJet] pseudojets
    cdef np.ndarray particles
    cdef int ievent = 0
    cdef object jets
    for particles in iterable:
        # convert numpy array into vector of pseudojets
        array_to_pseudojets(particles.shape[0], <DTYPE_t*> particles.data,
                            pseudojets, eta_max)
        jets = cluster_pseudojets(
            pseudojets,
            jet_size,
            subjet_size_fraction,
            subjet_pt_min_fraction,
            subjet_dr_min,
            trimmed_pt_min, trimmed_pt_max,
            trimmed_mass_min, trimmed_mass_max,
            shrink, shrink_mass,
            compute_auxvars)

        if (not skip_failed) or (jets is not None):
            yield jets
            ievent += 1
            if ievent == events:
                # early termination
                break


@cython.boundscheck(False)
@cython.wraparound(False)
def cluster_hepmc(HepMCInput hepmc_input,
                  float eta_max,
                  float jet_size,
                  float subjet_size_fraction,
                  float subjet_pt_min_fraction,
                  float subjet_dr_min,
                  float trimmed_pt_min, float trimmed_pt_max, 
                  float trimmed_mass_min, float trimmed_mass_max,
                  bool shrink, float shrink_mass,
                  bool compute_auxvars):
    cdef vector[PseudoJet] pseudojets
    # event loop 
    while hepmc_input.get_next_event():
        # convert generator output directly into pseudojets
        hepmc_input.to_pseudojet(pseudojets, eta_max)
        # yield jets
        yield cluster_pseudojets(
            pseudojets,
            jet_size,
            subjet_size_fraction,
            subjet_pt_min_fraction,
            subjet_dr_min,
            trimmed_pt_min, trimmed_pt_max,
            trimmed_mass_min, trimmed_mass_max,
            shrink, shrink_mass,
            compute_auxvars)
    hepmc_input.finish()


#cdef run_delphes(np.ndarray particles, string config,
                 #int random_state,
                 #string objects="Calorimeter/towers"):
    #"""
    #Reconstruct detector-level objects with Delphes
    #"""
    ## Delphes init
    #cdef ExRootConfReader* delphes_config_reader = NULL
    #cdef Delphes* modular_delphes = NULL
    #cdef TObjArray* delphes_all_particles = NULL
    #cdef TObjArray* delphes_stable_particles = NULL
    #cdef TObjArray* delphes_partons = NULL
    #cdef TObjArray* delphes_input_array = NULL
    
    #delphes_config_reader = new ExRootConfReader()
    #delphes_config_reader.ReadFile(config.c_str())
    ## Set Delhes' random seed. Only possible through a config file...
    #with tempfile.NamedTemporaryFile() as tmp:
        #tmp.write("set RandomSeed {0:d}\n".format(random_state))
        #tmp.flush()
        #delphes_config_reader.ReadFile(tmp.name)
    #modular_delphes = new Delphes("Delphes")
    #modular_delphes.SetConfReader(delphes_config_reader)
    #delphes_all_particles = modular_delphes.ExportArray("allParticles")
    #delphes_stable_particles = modular_delphes.ExportArray("stableParticles")
    #delphes_partons = modular_delphes.ExportArray("partons")
    #modular_delphes.InitTask()
    #delphes_input_array = modular_delphes.ImportArray(detector_objects)

    #cdef int ievent;
    #cdef Result* result
    #cdef vector[PseudoJet] particles
    
    #try:
                #detector_jets = None
                #modular_delphes.Clear()
                ## convert generator particles into Delphes candidates
                #gen_input.to_delphes(modular_delphes,
                                     #delphes_all_particles,
                                     #delphes_stable_particles,
                                     #delphes_partons)
                ## run Delphes reconstruction
                #modular_delphes.ProcessTask()
                ## convert Delphes candidates into pseudojets
                #delphes_to_pseudojet(delphes_input_array, particles)

                ## run jet clustering
                #result = get_jets(particles,
                                  #jet_size, subjet_size_fraction,
                                  #subjet_pt_min_fraction,
                                  #subjet_dr_min,
                                  #trimmed_pt_min, trimmed_pt_max,
                                  #trimmed_mass_min, trimmed_mass_max,
                                  #shrink, shrink_mass,
                                  #compute_auxvars)

                #if result != NULL:
                    #detector_jets = Jets()
                    #jets_from_result(detector_jets, result)
                    #del result

                #yield truth_jets, detector_jets

            #else:
                #yield truth_jets
            #ievent += 1
        #gen_input.finish()
    #finally:
        #del modular_delphes
        #del delphes_config_reader
