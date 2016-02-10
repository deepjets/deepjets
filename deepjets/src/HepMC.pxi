
cdef extern from "HepMC/GenEvent.h" namespace "HepMC":
    cdef cppclass GenEvent:
        pass


cdef extern from "HepMC/IO_GenEvent.h" namespace "HepMC":
    cdef cppclass IO_GenEvent:
        GenEvent* read_next_event()
