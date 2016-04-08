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
    array_to_pseudojets(particles.shape[0], len(particles.dtype.names),
                        <DTYPE_t*> particles.data,
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
        array_to_pseudojets(particles.shape[0], len(particles.dtype.names),
                            <DTYPE_t*> particles.data,
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
        array_to_pseudojets(particles.shape[0], len(particles.dtype.names),
                            <DTYPE_t*> particles.data,
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
def cluster_mc(MCInput gen_input,
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
    while gen_input.get_next_event():
        # convert generator output directly into pseudojets
        gen_input.to_pseudojet(pseudojets, eta_max)
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
    gen_input.finish()

