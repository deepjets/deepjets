// DIRE includes.
#include "DIRE/WeightContainer.h"

// Pythia includes.
#include "Pythia8/Pythia.h"

// Use UserHooks as trick to initialise the Weight container.

class WeightHooks : public Pythia8::UserHooks {

public:

  // Constructor and destructor.
  WeightHooks(Pythia8::WeightContainer* weightsIn) : weights(weightsIn) {}
 ~WeightHooks() {}

  bool canVetoProcessLevel() { return true;}
  bool doVetoProcessLevel( Pythia8::Event&) {
    weights->init();
    return false;
  }

  Pythia8::WeightContainer* weights;

};
