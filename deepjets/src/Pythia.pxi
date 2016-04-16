
cdef extern from "Pythia8/Pythia.h" namespace "Pythia8":
    cdef cppclass Event:
        pass

    cdef cppclass Pythia:
        Event event
        Pythia(string, bool)
        bool readString(string)
        bool readFile(string)
        bool init()
        bool next()
        void stat()

cdef extern from "Vincia/Vincia.h" namespace "Vincia":
    cdef cppclass VinciaPlugin:
        VinciaPlugin(Pythia*, string)
        void init()
