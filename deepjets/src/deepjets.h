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


void get_jets(Event& event,
              double etaMax,
              double R, double TR,
              double JpTMin, double TJpTMin) {
  // Find leading pT jet in event with anti-kt algorithm (params R, JpTMin, etaMax).
  // Find subjets by re-cluster jet using kt algorith (params TR, TJpTMin).
  // Write jet properties, constituents to fname_jets.csv, fname_csts.csv.

  // Set up FastJet jet finder.
  fastjet::JetDefinition jetDef(fastjet::genkt_algorithm, R, -1);
  std::vector <fastjet::PseudoJet> fjInputs;
  fjInputs.resize(0);

  // Begin FastJet analysis: extract particles from event record.
  Vec4 pTemp;
  for ( int i = 0; i < event.size(); ++i ) if ( event[i].isFinal() ) {
    // Require visible particles inside detector.
    if ( !event[i].isVisible() ) continue;
    if ( etaMax < 20. && abs(event[i].eta()) > etaMax ) continue;

    // Create a PseudoJet from the complete Pythia particle.
    fastjet::PseudoJet particleTemp = event[i];

    // Store acceptable particles as input to Fastjet.
    // Conversion to PseudoJet is performed automatically
    // with the help of the code in FastJet3.h.
    fjInputs.push_back( particleTemp );
  }

  // Run Fastjet algorithm and sort jets in pT order.
  vector <fastjet::PseudoJet> inclusiveJets, sortedJets;
  fastjet::ClusterSequence clustSeq( fjInputs, jetDef ); // Segmentation fault here sometimes.
  inclusiveJets = clustSeq.inclusive_jets( JpTMin );
  sortedJets    = sorted_by_pt(inclusiveJets);

  // Get details and constituents from leading jet.
  vector <fastjet::PseudoJet> Jconstits = sortedJets[0].constituents();
  double JpT  = sortedJets[0].perp();
  double Jeta = sortedJets[0].eta();
  double Jphi = sortedJets[0].phi_std();
  int Jsize   = Jconstits.size();

  // Output leading jet details.
  //fprintf( f_jets, "%g, %g, %g, %i\n", JpT, Jeta, Jphi, Jsize );

  // Set up FastJet jet trimmer.
  fastjet::JetDefinition TjetDef( fastjet::genkt_algorithm, TR, 1 );
  std::vector <fastjet::PseudoJet> TfjInputs;
  TfjInputs.resize(0);
  for ( int i = 0; i < Jsize; ++i ) {
    // Store constituents from leading jet.
    TfjInputs.push_back( Jconstits[i] );
  }

  // Run Fastjet trimmer on leading jet.
  vector <fastjet::PseudoJet> TinclusiveJets, TsortedJets;
  fastjet::ClusterSequence TclustSeq( TfjInputs, TjetDef );
  TinclusiveJets = TclustSeq.inclusive_jets( JpT*TJpTMin );
  TsortedJets    = sorted_by_pt(TinclusiveJets);

  // Get details and constituents from subjets.
  for ( int j = 0; j < int( TsortedJets.size() ); ++j ) {
    Jconstits = TsortedJets[j].constituents();
    JpT       = TsortedJets[j].perp();
    Jeta      = TsortedJets[j].eta();
    Jphi      = TsortedJets[j].phi_std();
    Jsize     = Jconstits.size();
    // Output subjet details.
    //fprintf( f_jets, "%g, %g, %g, %i\n", JpT, Jeta, Jphi, Jsize );
    for (int i = 0; i < Jsize; ++i) {
      double cE   = Jconstits[i].E();
      double cEt  = Jconstits[i].Et();
      double ceta = Jconstits[i].eta();
      double cphi = Jconstits[i].phi_std();
      // Output constituent details.
      //fprintf( f_csts, "%g, %g, %g, %g\n", cE, cEt, ceta, cphi );
    }
  }
}
