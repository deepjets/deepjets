#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/FastJet3.h"
#include <stdio.h>
#include <stdlib.h>

using namespace Pythia8;

void JetAnalysis( Event& event, const char* fname,
                  double R, double JpTMin, double etaMax, double TR, double TJpTMin ) {
  // Find leading pT jet in event with anti-kt algorithm (params R, JpTMin, etaMax).
  // Find subjets by re-cluster jet using kt algorith (params TR, TJpTMin).
  // Write jet properties, constituents to fname_jets.csv, fname_csts.csv.

  // Set up FastJet jet finder.
  fastjet::JetDefinition jetDef( fastjet::genkt_algorithm, R, -1 );
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
  
  // Open output files.
  char fjetsName[10+strlen(fname)];
  strcpy( fjetsName, fname );
  strcat( fjetsName, "_jets.csv" );
  FILE *f_jets = fopen(fjetsName, "w");
  char fcstsName[10+strlen(fname)];
  strcpy( fcstsName, fname );
  strcat( fcstsName, "_csts.csv" );
  FILE *f_csts = fopen(fcstsName, "w");

  // Output leading jet details.
  fprintf( f_jets, "%g, %g, %g, %i\n", JpT, Jeta, Jphi, Jsize );
  
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
    Jconstits = sortedJets[j].constituents();
    JpT       = TsortedJets[j].perp();
    Jeta      = TsortedJets[j].eta();
    Jphi      = TsortedJets[j].phi_std();
    Jsize     = Jconstits.size();
    // Output subjet details.
    fprintf( f_jets, "%g, %g, %g, %i\n", JpT, Jeta, Jphi, Jsize );
    for (int i = 0; i < Jsize; ++i) {
      double cE   = Jconstits[i].E();
      double cEt  = Jconstits[i].Et();
      double ceta = Jconstits[i].eta();
      double cphi = Jconstits[i].phi_std();
      // Output constituent details.
      fprintf( f_csts, "%g, %g, %g, %g\n", cE, cEt, ceta, cphi );
    }
    fprintf( f_csts, "\n" );
  }
  
  // Close output files.
  fclose( f_jets );
  fclose( f_csts );
}

int main() {
  // Generates W' events:
  // W' -> W Z; W -> q q', Z -> nu nu.
  // Performs jet analysis required to form jet images.
  // *** Need to specify location of Pythia xmldoc directory (xmldocDir)! ***

  // Main program settings.
  double eCM            = 13000.0;
  int nEvent            = 1;
  int nAbort            = 3;
  const char* xmldocDir = "/Users/barney800/Tools/Pythia8/share/Pythia8/xmldoc";
  // Parameters for FastJet analyses.
  double R          = 0.6;    // Jet size
  double JpTMin     = 12.5;   // Min jet pT
  double etaMax     = 5.0;    // Pseudorapidity range of detector
  double TR         = 0.3;    // Subjet size
  double TJpTMin    = 0.05;   // Min subjet pT (fraction of jet pT)
  const char* fname = "test"; // Filename for outputs.
  // Other parameters.
  double WpTMin  = 250.0;  // Min W pT
  double WpTMax  = 300.0;  // Max W pT

  // Generator. Shorthand for the event.
  Pythia pythia( xmldocDir );
  Event& event = pythia.event;
  
  // Use variable random seed: -1 = default, 0 = clock.
  pythia.settings.flag("Random:setSeed", "on");
  pythia.settings.mode("Random:seed", 0);

  // Set up beams: p p is default so only need set energy.
  pythia.settings.parm("Beams:eCM", eCM);

  // W' pair production.
  pythia.readString("NewGaugeBoson:ffbar2Wprime = on");
  pythia.readString("34:m0 = 700.0");
  
  // W' decays.
  pythia.readString("Wprime:coup2WZ = 1.0");
  pythia.readString("34:onMode = off");
  pythia.readString("34:onIfAny = 24");
  
  // W and Z decays.
  pythia.readString("24:onMode = off");
  pythia.readString("24:onIfAny = 1 2 3 4 5 6");
  pythia.readString("23:onMode = off");
  pythia.readString("23:onIfAny = 12 14 16");

  // Switch on/off particle data and event listings.
  pythia.readString("Init:showChangedSettings = off");
  pythia.readString("Init:showChangedParticleData = off");
  pythia.readString("Next:numberShowInfo = 1");
  pythia.readString("Next:numberShowProcess = 1");
  pythia.readString("Next:numberShowEvent = 0");

  // Initialize.
  pythia.init();
  
  // Begin event loop.
  int iAbort = 0;
  int iEvent = 0;
  while ( iEvent < nEvent ) {
    // Generate event. Quit if failure.
    if ( !pythia.next() ) {
      if ( ++iAbort < nAbort ) continue;
      cout << " Event generation aborted prematurely, owing to error.\n";
      break;
    }
    
    // Check W pT.
    int checkWpT = 0;
    double WpT   = 0.0;
    double Weta  = 0.0;
    double Wphi  = 0.0;
    for ( int i = 0; i < event.size(); ++i ) {
      if ( event[i].id() == 24 || event[i].id() == -24 ) {
        WpT = abs(event[i].pT());
        Weta = event[i].eta();
        Wphi = event[i].phi();
        if ( WpT < WpTMin || WpT > WpTMax ) {
          break;
        }
        else {
          checkWpT = 1;
          break;
        }
      }
    }
    if ( checkWpT == 0 ) continue;
    ++iEvent;
    
    // Perform jet analysis.
    JetAnalysis( event, fname, R, JpTMin, etaMax, TR, TJpTMin );
 
    // End of event loop.
  }

  // Done.
  return 0;
}
