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


bool keep_event(Event& event, double WpTMin, double WpTMax) {
  // Check W pT.
  if (WpTMin < 0 && WpTMax < 0) return true;
  int checkWpT = 0;
  double WpT   = 0.;
  for (int i = 0; i < event.size(); ++i) {
    if (event[i].id() == 24 || event[i].id() == -24) {
      WpT = abs(event[i].pT());
      if (WpT < WpTMin || WpT > WpTMax) {
        break;
      } else {
        checkWpT = 1;
        break;
      }
    }
  }
  return checkWpT != 0;
}

void jets_to_arrays(Result& result,
                    double* jet_arr, double* jet_constit_arr,
                    double* subjets_arr, double* subjets_constit_arr) {
  /*
   * Fill arrays from the contents of a Result struct
   */
  jet_arr[0] = result.jet.perp();
  jet_arr[1] = result.jet.eta();
  jet_arr[2] = result.jet.phi_std();

  std::vector<fastjet::PseudoJet> constits = result.jet.constituents();

  for (unsigned int i = 0; i < constits.size(); ++i) {
      jet_constit_arr[i * 4 + 0] = constits[i].E();
      jet_constit_arr[i * 4 + 1] = constits[i].Et();
      jet_constit_arr[i * 4 + 2] = constits[i].eta();
      jet_constit_arr[i * 4 + 3] = constits[i].phi_std();
  }

  // Get details and constituents from subjets.
  int iconstit = 0;
  for (unsigned int i = 0; i < result.subjets.size(); ++i) {
    subjets_arr[i * 3 + 0] = result.subjets[i].perp();
    subjets_arr[i * 3 + 1] = result.subjets[i].eta();
    subjets_arr[i * 3 + 2] = result.subjets[i].phi_std();
    constits = result.subjets[i].constituents();
    for (unsigned int j = 0; j < constits.size(); ++j) {
      subjets_constit_arr[iconstit * 4 + 0] = constits[j].E();
      subjets_constit_arr[iconstit * 4 + 1] = constits[j].Et();
      subjets_constit_arr[iconstit * 4 + 2] = constits[j].eta();
      subjets_constit_arr[iconstit * 4 + 3] = constits[j].phi_std();
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
