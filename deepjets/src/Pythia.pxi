
cdef extern from "Pythia8/Pythia.h" namespace "Pythia8":
    cdef cppclass Event:
        pass

    cdef cppclass Info:
        int nWeights()
        double weight(int)

    cdef cppclass Settings:
        void addFlag(string, bool) 
        void addMode(string, int, bool, bool, int, int)
        bool flag(string)
        int mode(string)
        double parm(string)

    cdef cppclass TimeShower:
        pass

    cdef cppclass SpaceShower:
        pass

    cdef cppclass UserHooks:
        pass

    cdef cppclass MergingHooks:
        pass

    cdef cppclass ParticleData:
        pass

    cdef cppclass Rndm:
        pass

    cdef cppclass PartonSystems:
        pass

    cdef cppclass BeamParticle:
        int id()
        double mQuarkPDF(int)

    cdef cppclass Pythia:
        Event event
        Info info
        Settings settings
        ParticleData particleData
        Rndm rndm
        PartonSystems partonSystems
        Pythia(string, bool)
        bool readString(string)
        bool readFile(string)
        bool init()
        bool next()
        void stat()
        bool setShowerPtr(TimeShower*, TimeShower*, SpaceShower*)
        bool setUserHooksPtr(UserHooks*)

cdef extern from "Vincia/Vincia.h" namespace "Vincia":
    cdef cppclass VinciaPlugin:
        VinciaPlugin(Pythia*, string)
        void init()

cdef extern from "DIRE/WeightContainer.h" namespace "Pythia8":
    cdef cppclass WeightContainer:
        void calcWeight(double)
        void reset()
        double getShowerWeight()

cdef extern from "DIRE/DireTimes.h" namespace "Pythia8":
    cdef cppclass DireTimes(TimeShower):
        DireTimes(Pythia*)
        void setWeightContainerPtr(WeightContainer*)
        void reinitPtr(Info*, Settings*, ParticleData*, Rndm*, PartonSystems*, UserHooks*, MergingHooks*, SplittingLibrary*) 

cdef extern from "DIRE/DireSpace.h" namespace "Pythia8":
    cdef cppclass DireSpace(SpaceShower):
        BeamParticle* beamAPtr
        BeamParticle* beamBPtr
        DireSpace(Pythia*)
        void setWeightContainerPtr(WeightContainer*)
        void reinitPtr(Info*, Settings*, ParticleData*, Rndm*, PartonSystems*, UserHooks*, MergingHooks*, SplittingLibrary*) 

cdef extern from "DIRE/SplittingLibrary.h" namespace "Pythia8":
    cdef cppclass SplittingLibrary:
       void init(Settings*, ParticleData*, Rndm*, BeamParticle*, BeamParticle*)
       void setTimesPtr(DireTimes*)
       void setSpacePtr(DireSpace*)

cdef extern from "pythia_dire_utils.h":
    cdef cppclass WeightHooks(UserHooks):
        WeightHooks(WeightContainer*)
