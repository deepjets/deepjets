


#include "HepMC/IO_GenEvent.h"
#include "HepMC/GenEvent.h"
#include <math.h>
#include <algorithm>
#include <vector>

/// Stores final state particles from file "input.hepmc"

/// \class  IsStateFinal
/// this predicate returns true if the input has no decay vertex
class IsStateFinal {
public:
    bool operator()( const HepMC::GenParticle* p ) {
        if ( !p->end_vertex() && p->status()==1 ) return 1;
        return 0;
    }
};

int main() {
	{
	HepMC::IO_GenEvent ascii_in("input.hepmc",std::ios::in);	
	HepMC::GenEvent* evt = ascii_in.read_next_event();
	IsStateFinal isfinal;
	std::vector<HepMC::GenParticle*> finalstateparticles;
	for ( HepMC::GenEvent::particle_iterator p = evt->particles_begin();
	      p != evt->particles_end(); ++p ) {
	    if ( isfinal(*p) )  finalstateparticles.push_back(*p);
	}
	}
	return 0;
}


