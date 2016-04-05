#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC2.h"
#include "fastjet/ClusterSequence.hh"

#include "HepMC/IO_GenEvent.h"
#include "HepMC/GenEvent.h"

#include "Delphes/modules/Delphes.h"
#include "Delphes/classes/DelphesClasses.h"
#include "Delphes/classes/DelphesFactory.h"

#include "TObjArray.h"
#include "TLorentzVector.h"

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include <math.h>
#include <vector>


class HepMC_IsStateFinal {
public:
  bool operator()( const HepMC::GenParticle* p ) {
    if ( !p->end_vertex() && p->status() == 1 ) return true;
    return false;
  }
};


HepMC::IO_GenEvent* get_hepmc_reader(std::string filename) {
  HepMC::IO_GenEvent* ascii_in = new HepMC::IO_GenEvent(filename, std::ios::in);
  return ascii_in;
}


void hepmc_finalstate_particles(HepMC::GenEvent* event, std::vector<HepMC::GenParticle*>& particles) {
  int pdgid;
  HepMC_IsStateFinal isfinal;
  particles.clear();
  for (HepMC::GenEvent::particle_iterator p = event->particles_begin(); p != event->particles_end(); ++p) if (isfinal(*p)) {
    // visibility test
    pdgid = abs((*p)->pdg_id());
    if ((pdgid == 12) || (pdgid == 14) || (pdgid == 16)) continue; // neutrino
    particles.push_back(*p);
  }
}


void hepmc_to_pseudojet(HepMC::GenEvent& evt, std::vector<fastjet::PseudoJet>& output, double eta_max) {
  int pdgid;
  HepMC_IsStateFinal isfinal;
  HepMC::FourVector fourvect;
  output.clear();
  for (HepMC::GenEvent::particle_iterator p = evt.particles_begin(); p != evt.particles_end(); ++p) if (isfinal(*p)) {
    // visibility test
    pdgid = abs((*p)->pdg_id());
    if ((pdgid == 12) || (pdgid == 14) || (pdgid == 16)) continue; // neutrino
    fourvect = (*p)->momentum();
    if (abs(fourvect.pseudoRapidity()) > eta_max) continue;
    fastjet::PseudoJet particleTemp(fourvect.px(), fourvect.py(), fourvect.pz(), fourvect.e());
    output.push_back(particleTemp);
  }
}


void pythia_to_pseudojet(Pythia8::Event& event, std::vector<fastjet::PseudoJet>& output, double eta_max) {
  output.clear();
  for (int i = 0; i < event.size(); ++i) if (event[i].isFinal()) {
    // Require visible particles inside detector.
    if (!event[i].isVisible()) continue;
    if (abs(event[i].eta()) > eta_max) continue;
    // Create a PseudoJet from the complete Pythia particle.
    fastjet::PseudoJet particleTemp = event[i];
    // TODO: associate particle to pseudojet with
    // fastjet::PseudoJet::UserInfoBase
    // see
    // http://fastjet.hepforge.org/svn/contrib/contribs/VertexJets/trunk/example.cc
    output.push_back(particleTemp);
  }
}


void pythia_to_delphes(Pythia8::Event& event, Delphes* delphes,
                       TObjArray* all_particles,
                       TObjArray* stable_particles,
                       TObjArray* partons) {
    // Based on code here:
    // https://cp3.irmp.ucl.ac.be/projects/delphes/browser/examples/ExternalFastJet/ExternalFastJetBasic.cpp
    DelphesFactory* factory = delphes->GetFactory();
    Candidate* candidate;
    int pdgid;
    for (int i = 0; i < event.size(); ++i) {
        if (!event[i].isVisible()) continue;
        candidate = factory->NewCandidate();
        const Pythia8::Particle& particle = event[i];
        candidate->PID = particle.id();
        pdgid = particle.idAbs();
        candidate->Status = particle.status();
        candidate->Charge = particle.charge();
        candidate->Mass = particle.m();
        candidate->Momentum.SetPxPyPzE(particle.px(), particle.py(), particle.pz(), particle.e());
        candidate->Position.SetXYZT(particle.xProd(), particle.yProd(), particle.zProd(), particle.tProd());
        all_particles->Add(candidate);
        if (particle.isFinal()) {
            stable_particles->Add(candidate);
        } else if (pdgid <= 5 || pdgid == 21 || pdgid == 15) {
            partons->Add(candidate);
        }
    }
}


void delphes_to_pseudojet(TObjArray* input_array, std::vector<fastjet::PseudoJet>& output) {
    // Based on code here:
    // https://cp3.irmp.ucl.ac.be/projects/delphes/browser/examples/ExternalFastJet/ExternalFastJetBasic.cpp
    output.clear();
    TIterator* input_iterator = input_array->MakeIterator();
    Candidate* candidate;
    fastjet::PseudoJet pseudojet;
    TLorentzVector momentum;
    while((candidate = static_cast<Candidate*>(input_iterator->Next()))) {
        momentum = candidate->Momentum;
        pseudojet = fastjet::PseudoJet(momentum.Px(), momentum.Py(), momentum.Pz(), momentum.E());
        // TODO: associate Candidate to PseudoJet with a
        // fastjet::PseudoJet::UserInfoBase
        // see
        // http://fastjet.hepforge.org/svn/contrib/contribs/VertexJets/trunk/example.cc
        output.push_back(pseudojet);
    }
}


HepMC::GenEvent* pythia_to_hepmc(Pythia8::Pythia* pythia) {
    HepMC::Pythia8ToHepMC py2hepmc;
    HepMC::GenEvent* event = new HepMC::GenEvent();
    if (!py2hepmc.fill_next_event(*pythia, event)) {
        delete event;
        return NULL;
    }
    return event;
}

void particles_to_array(std::vector<HepMC::GenParticle*>& particles, double* array) {
  HepMC::GenParticle* particle;
  HepMC::FourVector momentum, prod_vertex;
  for (unsigned int i = 0; i < particles.size(); ++i) {
    particle = particles[i];
    momentum = particle->momentum();
    prod_vertex = particle->production_vertex()->position();
    array[i * 10 + 0] = momentum.e();
    array[i * 10 + 1] = momentum.px();
    array[i * 10 + 2] = momentum.py();
    array[i * 10 + 3] = momentum.pz();
    array[i * 10 + 4] = momentum.m();
    array[i * 10 + 5] = prod_vertex.x();
    array[i * 10 + 6] = prod_vertex.y();
    array[i * 10 + 7] = prod_vertex.z();
    array[i * 10 + 8] = prod_vertex.t();
    array[i * 10 + 9] = particle->pdg_id();
  }
}
