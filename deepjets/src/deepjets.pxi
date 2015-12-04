
cdef extern from "deepjets.h":
    bool keep_event(Event&, double, double)
    void get_jets(Event&, double, double, double, double, double)
