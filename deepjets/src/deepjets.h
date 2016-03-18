#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/FastJet3.h"

#include "fastjet/contrib/Nsubjettiness.hh"

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


using namespace Pythia8;

class IsStateFinal {
public:
  bool operator()( const HepMC::GenParticle* p ) {
    if ( !p->end_vertex() && p->status()==1 ) return 1;
    return 0;
  }
};


HepMC::IO_GenEvent* get_hepmc_reader(std::string filename) {
  HepMC::IO_GenEvent* ascii_in = new HepMC::IO_GenEvent(filename, std::ios::in);
  return ascii_in;
}


void hepmc_to_pseudojet(HepMC::GenEvent& evt, std::vector<fastjet::PseudoJet>& output, double eta_max) {
  int pdgid;
  IsStateFinal isfinal;
  HepMC::FourVector fourvect;
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
  for (int i = 0; i < event.size(); ++i) if (event[i].isFinal()) {
    // Require visible particles inside detector.
    if (!event[i].isVisible()) continue;
    if (abs(event[i].eta()) > eta_max) continue;
    // Create a PseudoJet from the complete Pythia particle.
    fastjet::PseudoJet particleTemp = event[i];
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
    TIterator* input_iterator = input_array->MakeIterator();
    Candidate* candidate;
    fastjet::PseudoJet pseudojet;
    TLorentzVector momentum;
    while((candidate = static_cast<Candidate*>(input_iterator->Next()))) {
        momentum = candidate->Momentum;
        pseudojet = fastjet::PseudoJet(momentum.Px(), momentum.Py(), momentum.Pz(), momentum.E());
        output.push_back(pseudojet);
    }
}


/*
 * Instead if passing around many arguments, we use a Result struct
 */
struct Result {
  ~Result() {
    delete jet_clusterseq;
    delete subjet_clusterseq;
  }
  fastjet::PseudoJet jet;
  fastjet::PseudoJet trimmed_jet;
  std::vector<fastjet::PseudoJet> subjets;
  fastjet::ClusterSequence* jet_clusterseq;
  fastjet::ClusterSequence* subjet_clusterseq;
  double shrinkage;
  double subjet_dr;
  double tau_1;
  double tau_2;
  double tau_3;
};


bool keep_pythia_event(Pythia8::Event& event, int cut_on_pdgid, double pt_min, double pt_max) {
  if (pt_min < 0 && pt_max < 0) return true;
  bool passes = false;
  double pt;
  for (int i = 0; i < event.size(); ++i) {
    if (abs(event[i].id()) == cut_on_pdgid) {
      pt = abs(event[i].pT());
      if (pt_min >= 0 && pt < pt_min) {
        break;
      } else if (pt_max > pt_min && pt > pt_max) {
        break;
      }
      passes = true;
      break;
    }
  }
  return passes;
}


void result_to_arrays(Result& result,
                      double* jet_arr, double* subjet_arr,
                      double* jet_constit_arr, double* subjet_constit_arr) {
  /*
   * Fill arrays from the contents of a Result struct
   */
  // The original jet
  jet_arr[0] = result.jet.perp();
  jet_arr[1] = result.jet.eta();
  jet_arr[2] = result.jet.phi_std();
  jet_arr[3] = result.jet.m();
  // The trimmed jet
  jet_arr[4] = result.trimmed_jet.perp();
  jet_arr[5] = result.trimmed_jet.eta();
  jet_arr[6] = result.trimmed_jet.phi_std();
  jet_arr[7] = result.trimmed_jet.m();

  std::vector<fastjet::PseudoJet> constits = result.jet.constituents();

  for (unsigned int i = 0; i < constits.size(); ++i) {
    jet_constit_arr[i * 3 + 0] = constits[i].Et();
    jet_constit_arr[i * 3 + 1] = constits[i].eta();
    jet_constit_arr[i * 3 + 2] = constits[i].phi_std();
  }

  // Get details and constituents from subjets.
  int iconstit = 0;
  for (unsigned int i = 0; i < result.subjets.size(); ++i) {
    subjet_arr[i * 4 + 0] = result.subjets[i].perp();
    subjet_arr[i * 4 + 1] = result.subjets[i].eta();
    subjet_arr[i * 4 + 2] = result.subjets[i].phi_std();
    subjet_arr[i * 4 + 3] = result.subjets[i].m();
    constits = result.subjets[i].constituents();
    for (unsigned int j = 0; j < constits.size(); ++j) {
      subjet_constit_arr[iconstit * 3 + 0] = constits[j].Et();
      subjet_constit_arr[iconstit * 3 + 1] = constits[j].eta();
      subjet_constit_arr[iconstit * 3 + 2] = constits[j].phi_std();
      ++iconstit;
    }
  }
}


Result* get_jets(std::vector<fastjet::PseudoJet>& fjInputs,
                 double jet_size, double subjet_size_fraction,
                 double subjet_pt_min_fraction,
                 double subjet_dr_min,
                 double trimmed_pt_min, double trimmed_pt_max,
                 bool shrink, double shrink_mass,
                 bool compute_auxvars) {
  /*
   * Find leading pT jet in event with anti-kt algorithm (params jet_size, jet_pt_min).
   * Find subjets by re-cluster jet using kt algorith (params subjet_size_fraction * jet_size, subjet_size_fraction).
   * Return a Result struct
   */

  // Run Fastjet algorithm and sort jets in pT order
  fastjet::JetDefinition jetDef(fastjet::genkt_algorithm, jet_size, -1); // anti-kt
  fastjet::ClusterSequence* clustSeq = new fastjet::ClusterSequence(fjInputs, jetDef);
  std::vector<fastjet::PseudoJet> sortedjets(sorted_by_pt(clustSeq->inclusive_jets())); // no pT cut here

  if (sortedjets.empty()) {
    delete clustSeq;
    return NULL;
  }

  // Get leading jet
  fastjet::PseudoJet& jet = sortedjets[0];
  std::vector<fastjet::PseudoJet> Jconstits = jet.constituents();
  int Jsize = Jconstits.size();

  // Store constituents from leading jet
  std::vector<fastjet::PseudoJet> TfjInputs;
  for (int i = 0; i < Jsize; ++i) {
    TfjInputs.push_back(Jconstits[i]);
  }

  double shrinkage = 1.;
  double actual_size;
  if (shrink) {
    // Shrink distance parameter to 2 * m / pT
    if (shrink_mass <= 0) {
      // Use jet mass
      shrink_mass = jet.m();
      if (shrink_mass <= 0) {
        // Skip event
        delete clustSeq;
        return NULL;
      }
    }
    actual_size = 2 * shrink_mass / jet.perp();
    if (actual_size > jet_size) {
      // Original clustering must have been too small?
      // Skip event
      delete clustSeq;
      return NULL;
    }
    shrinkage = actual_size / jet_size;
    jet_size = actual_size;
  }

  // Run Fastjet trimmer on leading jet
  fastjet::JetDefinition TjetDef(fastjet::genkt_algorithm, subjet_size_fraction * jet_size, 1); // kt
  fastjet::ClusterSequence* TclustSeq = new fastjet::ClusterSequence(TfjInputs, TjetDef);
  std::vector<fastjet::PseudoJet> sortedsubjets(sorted_by_pt(TclustSeq->inclusive_jets(jet.perp() * subjet_pt_min_fraction)));

  // Sum subjets to make trimmed jet
  fastjet::PseudoJet trimmed_jet;
  for(std::vector<fastjet::PseudoJet>::iterator it = sortedsubjets.begin(); it != sortedsubjets.end(); ++it) {
    trimmed_jet += *it;
  }

  // pT cuts on trimmed jet
  if (trimmed_pt_min > 0) {
    if (trimmed_jet.perp() < trimmed_pt_min) {
      delete clustSeq;
      delete TclustSeq;
      return NULL;
    }
  }
  if (trimmed_pt_max > 0) {
    if (trimmed_jet.perp() >= trimmed_pt_max) {
      delete clustSeq;
      delete TclustSeq;
      return NULL;
    }
  }

  // Check subjet_dr_min condition
  double subjet_dr = -1;
  if (sortedsubjets.size() >= 2) {
    subjet_dr = sortedsubjets[0].delta_R(sortedsubjets[1]);
    if (subjet_dr_min > 0 && subjet_dr < subjet_dr_min) {
      // Skip event
      delete clustSeq;
      delete TclustSeq;
      return NULL;
    }
  }

  Result* result = new Result();
  result->jet = jet;
  result->trimmed_jet = trimmed_jet;
  result->subjets = sortedsubjets;
  result->jet_clusterseq = clustSeq;
  result->subjet_clusterseq = TclustSeq;
  result->shrinkage = shrinkage;
  result->subjet_dr = subjet_dr;

  if (compute_auxvars) {
    // Compute n-subjetiness ratio tau_21
    fastjet::contrib::NormalizedCutoffMeasure normalized_measure(1, jet_size, 1000000);
    fastjet::contrib::WTA_KT_Axes wta_kt_axes;
    fastjet::contrib::Nsubjettiness tau1(1, wta_kt_axes, normalized_measure);
    fastjet::contrib::Nsubjettiness tau2(2, wta_kt_axes, normalized_measure);
    fastjet::contrib::Nsubjettiness tau3(3, wta_kt_axes, normalized_measure);

    result->tau_1 = tau1.result(jet);
    result->tau_2 = tau2.result(jet);
    result->tau_3 = tau3.result(jet);
  }
  return result;
}
