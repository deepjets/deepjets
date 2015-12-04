#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/FastJet3.h"
#include <stdio.h>
#include <stdlib.h>

using namespace Pythia8;


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


void get_jets(Event& event, double* jet, double* subjets, double* constituents,
              int& num_subjets, int& num_constituents,
              double etaMax,
              double R, double TR,
              double JpTMin, double TJpTMin) {
  // Find leading pT jet in event with anti-kt algorithm (params R, JpTMin, etaMax).
  // Find subjets by re-cluster jet using kt algorith (params TR, TJpTMin).
  // Write jet properties, constituents to fname_jets.csv, fname_csts.csv.

  // Set up FastJet jet finder.
  fastjet::JetDefinition jetDef(fastjet::genkt_algorithm, R, -1);
  std::vector<fastjet::PseudoJet> fjInputs;
  fjInputs.resize(0);

  // Begin FastJet analysis: extract particles from event record.
  for (int i = 0; i < event.size(); ++i) if (event[i].isFinal()) {
    // Require visible particles inside detector.
    if ( !event[i].isVisible() ) continue;
    if ( etaMax < 20. && abs(event[i].eta()) > etaMax ) continue;

    // Create a PseudoJet from the complete Pythia particle.
    fastjet::PseudoJet particleTemp = event[i];

    // Store acceptable particles as input to Fastjet.
    // Conversion to PseudoJet is performed automatically
    // with the help of the code in FastJet3.h.
    fjInputs.push_back(particleTemp);
  }

  // Run Fastjet algorithm and sort jets in pT order.
  vector<fastjet::PseudoJet> inclusiveJets, sortedJets;
  fastjet::ClusterSequence clustSeq(fjInputs, jetDef);
  inclusiveJets = clustSeq.inclusive_jets(JpTMin);
  sortedJets    = sorted_by_pt(inclusiveJets);

  // Get details and constituents from leading jet.
  fastjet::PseudoJet& leading_jet = sortedJets[0];
  std::vector<fastjet::PseudoJet> Jconstits = leading_jet.constituents();
  double JpT = leading_jet.perp();
  jet[0] = JpT;
  jet[1] = leading_jet.eta();
  jet[2] = leading_jet.phi_std();
  int Jsize = Jconstits.size();

  // Set up FastJet jet trimmer.
  fastjet::JetDefinition TjetDef(fastjet::genkt_algorithm, TR, 1);
  std::vector<fastjet::PseudoJet> TfjInputs;
  for (int i = 0; i < Jsize; ++i) {
    // Store constituents from leading jet.
    TfjInputs.push_back(Jconstits[i]);
  }

  // Run Fastjet trimmer on leading jet.
  std::vector<fastjet::PseudoJet> TinclusiveJets, TsortedJets;
  fastjet::ClusterSequence TclustSeq(TfjInputs, TjetDef);
  TinclusiveJets = TclustSeq.inclusive_jets(JpT * TJpTMin);
  TsortedJets = sorted_by_pt(TinclusiveJets);

  num_subjets = TsortedJets.size();
  subjets = new double[TsortedJets.size() * 3];
  constituents = new double[Jsize * 4];

  double cE, cEt, ceta, cphi;
  int iconstit = 0;

  // Get details and constituents from subjets.
  for (int j = 0; j < int(TsortedJets.size()); ++j) {
    Jconstits = TsortedJets[j].constituents();
    Jsize     = Jconstits.size();
    subjets[j * 3 + 0] = TsortedJets[j].perp();
    subjets[j * 3 + 1] = TsortedJets[j].eta();
    subjets[j * 3 + 2] = TsortedJets[j].phi_std();
    // Output subjet details.
    //fprintf( f_jets, "%g, %g, %g, %i\n", JpT, Jeta, Jphi, Jsize );
    for (int i = 0; i < Jsize; ++i) {
      constituents[iconstit * 4 + 0] = Jconstits[i].E();
      constituents[iconstit * 4 + 1] = Jconstits[i].Et();
      constituents[iconstit * 4 + 2] = Jconstits[i].eta();
      constituents[iconstit * 4 + 3] = Jconstits[i].phi_std();
      ++iconstit;
    }
  }
  num_constituents = iconstit;
}
