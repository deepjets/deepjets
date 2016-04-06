import os


cdef class GeneratorInput:
    cdef bool get_next_event(self) except *:
        return False
    
    cdef GenEvent* get_hepmc(self):
        pass

    cdef void to_pseudojet(self, vector[PseudoJet]& particles, float eta_max):
        pass

    cdef void to_delphes(self, Delphes* modular_delphes,
                         TObjArray* delphes_all_particles,
                         TObjArray* delphes_stable_particles,
                         TObjArray* delphes_partons):
        pass

    cdef void finish(self):
        pass


cdef class PythiaInput(GeneratorInput):
    cdef Pythia* pythia
    cdef int cut_on_pdgid
    cdef float pdgid_pt_min
    cdef float pdgid_pt_max
    cdef int verbosity

    def __cinit__(self, string config, string xmldoc,
                  int random_state=0, float beam_ecm=13000.,
                  int cut_on_pdgid=0,
                  float pdgid_pt_min=-1, float pdgid_pt_max=-1,
                  object params_dict=None, int verbosity=1):
        self.pythia = new Pythia(xmldoc, False)
        if verbosity > 0:
            self.pythia.readString("Init:showProcesses = on")
            self.pythia.readString("Init:showChangedSettings = on")
        else:
            self.pythia.readString("Init:showProcesses = off")
            self.pythia.readString("Init:showChangedSettings = off")
        if verbosity > 1:
            self.pythia.readString("Init:showMultipartonInteractions = on")
            self.pythia.readString("Init:showChangedParticleData = on")
            self.pythia.readString("Next:numberShowInfo = 1")
            self.pythia.readString("Next:numberShowProcess = 1")
            self.pythia.readString("Next:numberShowEvent = 1")
        else:
            self.pythia.readString("Init:showMultipartonInteractions = off")
            self.pythia.readString("Init:showChangedParticleData = off")
            self.pythia.readString("Next:numberShowInfo = 0")
            self.pythia.readString("Next:numberShowProcess = 0")
            self.pythia.readString("Next:numberShowEvent = 0")
        self.pythia.readFile(config)  # read user config after options above
        self.pythia.readString('Beams:eCM = {0}'.format(beam_ecm))
        self.pythia.readString('Random:setSeed = on')
        self.pythia.readString('Random:seed = {0}'.format(random_state))
        if params_dict is not None:
            for param, value in params_dict.items():
                self.pythia.readString('{0} = {1}'.format(param, value))
        self.pythia.init()
        self.cut_on_pdgid = cut_on_pdgid
        self.pdgid_pt_min = pdgid_pt_min
        self.pdgid_pt_max = pdgid_pt_max
        self.verbosity = verbosity

    def __dealloc__(self):
        del self.pythia

    cdef bool get_next_event(self):
        # generate event and quit if failure
        if not self.pythia.next():
            raise RuntimeError("event generation aborted prematurely")
        
        if not keep_pythia_event(self.pythia.event, self.cut_on_pdgid,
                                 self.pdgid_pt_min, self.pdgid_pt_max):
            # event doesn't pass our truth-level cuts
            return False
        return True
    
    cdef GenEvent* get_hepmc(self):
        return pythia_to_hepmc(self.pythia)

    cdef void to_pseudojet(self, vector[PseudoJet]& particles, float eta_max):
        pythia_to_pseudojet(self.pythia.event, particles, eta_max)
        
    cdef void to_delphes(self, Delphes* modular_delphes,
                         TObjArray* delphes_all_particles,
                         TObjArray* delphes_stable_particles,
                         TObjArray* delphes_partons):
        # convert Pythia particles into Delphes candidates
        pythia_to_delphes(self.pythia.event, modular_delphes,
                          delphes_all_particles,
                          delphes_stable_particles,
                          delphes_partons)

    cdef void finish(self):
        if self.verbosity > 0:
            self.pythia.stat()


cdef class HepMCInput(GeneratorInput):
    cdef string filename
    cdef IO_GenEvent* hepmc_reader
    cdef GenEvent* event

    def __cinit__(self, string filename):
        self.filename = filename
        self.hepmc_reader = get_hepmc_reader(filename)
        self.event = NULL

    def __dealloc__(self):
        del self.event
        del self.hepmc_reader

    cdef bool get_next_event(self):
        self.event = self.hepmc_reader.read_next_event()
        if self.event == NULL:
            return False
        return True
    
    cdef GenEvent* get_hepmc(self):
        return self.event

    cdef void to_pseudojet(self, vector[PseudoJet]& particles, float eta_max):
        hepmc_to_pseudojet(self.event[0], particles, eta_max)
        del self.event
        self.event = NULL

    def estimate_num_events(self, int sample_size=1000):
        """
        Getting the exact number of events in a HepMC file is too expensive
        since this involves counting total number of lines beginning with "E"
        and the file can be GB in size. Even "wc -l file.hepmc" takes forever.
        Instead we can estimate the number of events by averaging the event
        sizes for the first N events and then dividing the total file size by
        that. This estimate may be used to report a progress bar as the reader
        loops over events.
        """
        cdef np.ndarray sizes = np.empty(sample_size, dtype=np.int32)
        cdef int num_found = 0
        cdef long long prev_location = 0
        filesize = os.path.getsize(self.filename)
        with open(self.filename, 'r') as infile:
            for line in infile:
                if line[0] == 'E':
                    if num_found > 0:
                        sizes[num_found - 1] = infile.tell() - prev_location
                    if num_found == sample_size:
                        break
                    num_found += 1
                    prev_location = infile.tell()
            else:
                return num_found
        return long(filesize / np.average(sizes))
