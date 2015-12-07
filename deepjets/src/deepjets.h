#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/FastJet3.h"
#include <stdio.h>
#include <stdlib.h>

using namespace Pythia8;

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
  double Weta  = 0.;
  double Wphi  = 0.;
  for (int i = 0; i < event.size(); ++i) {
    if (event[i].id() == 24 || event[i].id() == -24) {
      WpT = abs(event[i].pT());
      Weta = event[i].eta();
      Wphi = event[i].phi();
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

void jets_to_arrays(Result& result, double* jet_arr, double* subjets_arr, double* constit_arr) {
  jet_arr[0] = result.jet.perp();
  jet_arr[1] = result.jet.eta();
  jet_arr[2] = result.jet.phi_std();

  // Get details and constituents from subjets.
  int iconstit = 0;
  unsigned int Jsize;
  std::vector<fastjet::PseudoJet> Jconstits;

  for (unsigned int j = 0; j < result.subjets.size(); ++j) {
    Jconstits = result.subjets[j].constituents();
    Jsize     = Jconstits.size();
    subjets_arr[j * 3 + 0] = result.subjets[j].perp();
    subjets_arr[j * 3 + 1] = result.subjets[j].eta();
    subjets_arr[j * 3 + 2] = result.subjets[j].phi_std();
    // Output subjet details.
    //fprintf( f_jets, "%g, %g, %g, %i\n", JpT, Jeta, Jphi, Jsize );
    for (unsigned int i = 0; i < Jsize; ++i) {
      constit_arr[iconstit * 4 + 0] = Jconstits[i].E();
      constit_arr[iconstit * 4 + 1] = Jconstits[i].Et();
      constit_arr[iconstit * 4 + 2] = Jconstits[i].eta();
      constit_arr[iconstit * 4 + 3] = Jconstits[i].phi_std();
      ++iconstit;
    }
  }
}

void get_jets(Event& event,
              Result& result,
              double etaMax,
              double R, double TR,
              double JpTMin, double TJpTMin) {
  // Find leading pT jet in event with anti-kt algorithm (params R, JpTMin, etaMax).
  // Find subjets by re-cluster jet using kt algorith (params TR, TJpTMin).
  // Write jet properties, constituents to fname_jets.csv, fname_csts.csv.

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

  result.jet = jet;
  result.subjets = sorted_by_pt(TclustSeq->inclusive_jets(jet.perp() * TJpTMin));
  result.jet_clusterseq = clustSeq;
  result.subjet_clusterseq = TclustSeq;
}
