#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/FastJet3.h"
#include <stdio.h>
#include <stdlib.h>

using namespace Pythia8;

/*
 * Instead if passing around many arguments, we use a Result struct
 */
struct Result {
  ~Result() {
    delete jet_clusterseq;
    delete subjet_clusterseq;
  }
  fastjet::PseudoJet jet;
  std::vector<fastjet::PseudoJet> subjets;
  fastjet::ClusterSequence* jet_clusterseq;
  fastjet::ClusterSequence* subjet_clusterseq;
};


bool keep_event(Event& event, int cut_on_pdgid, double pt_min, double pt_max) {
  // Check W pT.
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

void jets_to_arrays(Result& result,
                    double* jet_arr, double* jet_constit_arr, double* subjet_constit_arr) {
  /*
   * Fill arrays from the contents of a Result struct
   */
  jet_arr[0] = result.jet.perp();
  jet_arr[1] = result.jet.eta();
  jet_arr[2] = result.jet.phi_std();

  std::vector<fastjet::PseudoJet> constits = result.jet.constituents();

  for (unsigned int i = 0; i < constits.size(); ++i) {
      jet_constit_arr[i * 3 + 0] = constits[i].Et();
      jet_constit_arr[i * 3 + 1] = constits[i].eta();
      jet_constit_arr[i * 3 + 2] = constits[i].phi_std();
  }

  // Get details and constituents from subjets.
  int iconstit = 0;
  for (unsigned int i = 0; i < result.subjets.size(); ++i) {
    jet_arr[(i + 1) * 3 + 0] = result.subjets[i].perp();
    jet_arr[(i + 1) * 3 + 1] = result.subjets[i].eta();
    jet_arr[(i + 1) * 3 + 2] = result.subjets[i].phi_std();
    constits = result.subjets[i].constituents();
    for (unsigned int j = 0; j < constits.size(); ++j) {
      subjet_constit_arr[iconstit * 3 + 0] = constits[j].Et();
      subjet_constit_arr[iconstit * 3 + 1] = constits[j].eta();
      subjet_constit_arr[iconstit * 3 + 2] = constits[j].phi_std();
      ++iconstit;
    }
  }
}

Result* get_jets(Event& event,
                 double etaMax,
                 double R, double TR,
                 double JpTMin, double TJpTMin) {
  /*
   * Find leading pT jet in event with anti-kt algorithm (params R, JpTMin, etaMax).
   * Find subjets by re-cluster jet using kt algorith (params TR, TJpTMin).
   * Return a Result struct
   */

  // Set up FastJet jet finder.
  fastjet::JetDefinition jetDef(fastjet::genkt_algorithm, R, -1); // anti-kt
  std::vector<fastjet::PseudoJet> fjInputs;

  // Begin FastJet analysis: extract particles from event record.
  for (int i = 0; i < event.size(); ++i) if (event[i].isFinal()) {
    // Require visible particles inside detector.
    if ( !event[i].isVisible() ) continue;
    if ( abs(event[i].eta()) > etaMax ) continue;

    // Create a PseudoJet from the complete Pythia particle.
    fastjet::PseudoJet particleTemp = event[i];

    // Store acceptable particles as input to Fastjet.
    // Conversion to PseudoJet is performed automatically
    // with the help of the code in FastJet3.h.
    fjInputs.push_back(particleTemp);
  }

  // Run Fastjet algorithm and sort jets in pT order.
  vector<fastjet::PseudoJet> sortedjets;
  fastjet::ClusterSequence* clustSeq = new fastjet::ClusterSequence(fjInputs, jetDef);
  sortedjets = sorted_by_pt(clustSeq->inclusive_jets(JpTMin));

  // Get details and constituents from leading jet.
  fastjet::PseudoJet& jet = sortedjets[0];
  std::vector<fastjet::PseudoJet> Jconstits = jet.constituents();
  int Jsize = Jconstits.size();

  // Set up FastJet jet trimmer.
  fastjet::JetDefinition TjetDef(fastjet::genkt_algorithm, TR, 1); // kt
  std::vector<fastjet::PseudoJet> TfjInputs;
  for (int i = 0; i < Jsize; ++i) {
    // Store constituents from leading jet.
    TfjInputs.push_back(Jconstits[i]);
  }

  // Run Fastjet trimmer on leading jet.
  fastjet::ClusterSequence* TclustSeq = new fastjet::ClusterSequence(TfjInputs, TjetDef);

  Result* result = new Result();
  result->jet = jet;
  result->subjets = sorted_by_pt(TclustSeq->inclusive_jets(jet.perp() * TJpTMin));
  result->jet_clusterseq = clustSeq;
  result->subjet_clusterseq = TclustSeq;
  return result;
}
