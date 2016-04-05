#include "fastjet/ClusterSequence.hh"
#include "fastjet/tools/Filter.hh"
#include "fastjet/contrib/Nsubjettiness.hh"

#include <vector>


/*
 * Instead if passing around many arguments...
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


Result* get_jets(std::vector<fastjet::PseudoJet>& jet_inputs,
                 double jet_size, double subjet_size_fraction,
                 double subjet_pt_min_fraction,
                 double subjet_dr_min,
                 double trimmed_pt_min, double trimmed_pt_max,
                 double trimmed_mass_min, double trimmed_mass_max,
                 bool shrink, double shrink_mass,
                 bool compute_auxvars) {
  /*
   * Find leading pT jet in event with anti-kt algorithm (params jet_size, jet_pt_min).
   * Find subjets by re-cluster jet using kt algorith (params subjet_size_fraction * jet_size, subjet_size_fraction).
   * Return a Result struct
   */

  double original_jet_size = jet_size;

  // Run Fastjet algorithm and sort jets in pT order
  fastjet::JetDefinition jet_def(fastjet::genkt_algorithm, jet_size, -1); // anti-kt
  fastjet::ClusterSequence* cluster_sequence = new fastjet::ClusterSequence(jet_inputs, jet_def);
  std::vector<fastjet::PseudoJet> sorted_jets(sorted_by_pt(cluster_sequence->inclusive_jets())); // no pT cut here

  if (sorted_jets.empty()) {
    delete cluster_sequence;
    return NULL;
  }

  // Get leading jet
  fastjet::PseudoJet& jet = sorted_jets[0];

  //// Store constituents from leading jet
  //int jet_size = jet_constituents.size();
  //std::vector<fastjet::PseudoJet> trimmed_jet_inputs;
  //for (int i = 0; i < jet_size; ++i) {
    //trimmed_jet_inputs.push_back(jet_constituents[i]);
  /*}*/

  double shrinkage = 1.;
  double actual_size;
  if (shrink) {
    // Shrink distance parameter to 2 * m / pT
    if (shrink_mass <= 0) {
      // Use jet mass
      shrink_mass = jet.m();
      if (shrink_mass <= 0) {
        // Skip event
        delete cluster_sequence;
        return NULL;
      }
    }
    actual_size = 2 * shrink_mass / jet.perp();
    if (actual_size > jet_size) {
      // Original clustering must have been too small?
      // Skip event
      delete cluster_sequence;
      return NULL;
    }
    shrinkage = actual_size / jet_size;
    jet_size = actual_size;
  }

  fastjet::Filter filter(fastjet::JetDefinition(fastjet::genkt_algorithm, subjet_size_fraction * jet_size, 1), // kt
                         fastjet::SelectorPtFractionMin(subjet_pt_min_fraction));
  fastjet::PseudoJet trimmed_jet = filter(jet);
  std::vector<fastjet::PseudoJet> sorted_subjets(sorted_by_pt(trimmed_jet.pieces()));

  // pT cuts on trimmed jet
  if (trimmed_pt_min > 0) {
    if (trimmed_jet.perp() < trimmed_pt_min) {
      delete cluster_sequence;
      //delete trimmed_cluster_sequence;
      return NULL;
    }
  }
  if (trimmed_pt_max > 0) {
    if (trimmed_jet.perp() >= trimmed_pt_max) {
      delete cluster_sequence;
      //delete trimmed_cluster_sequence;
      return NULL;
    }
  }

  // mass cuts on trimmed jet
  if (trimmed_mass_min > 0) {
    if (trimmed_jet.m() < trimmed_mass_min) {
      delete cluster_sequence;
      //delete trimmed_cluster_sequence;
      return NULL;
    }
  }
  if (trimmed_mass_max > 0) {
    if (trimmed_jet.m() >= trimmed_mass_max) {
      delete cluster_sequence;
      //delete trimmed_cluster_sequence;
      return NULL;
    }
  }

  // Check subjet_dr_min condition
  double subjet_dr = -1;
  if (sorted_subjets.size() >= 2) {
    subjet_dr = sorted_subjets[0].delta_R(sorted_subjets[1]);
    if (subjet_dr_min > 0 && subjet_dr < subjet_dr_min) {
      // Skip event
      delete cluster_sequence;
      //delete trimmed_cluster_sequence;
      return NULL;
    }
  }

  Result* result = new Result();
  result->jet = jet;
  result->trimmed_jet = trimmed_jet;
  result->subjets = sorted_subjets;
  result->jet_clusterseq = cluster_sequence;
  //result->subjet_clusterseq = trimmed_cluster_sequence;
  result->shrinkage = shrinkage;
  result->subjet_dr = subjet_dr;

  if (compute_auxvars) {
    // Compute n-subjetiness ratio tau_21
    fastjet::contrib::NormalizedCutoffMeasure normalized_measure(1, original_jet_size, 1000000);
    fastjet::contrib::WTA_KT_Axes wta_kt_axes;
    fastjet::contrib::Nsubjettiness tau1(1, wta_kt_axes, normalized_measure);
    fastjet::contrib::Nsubjettiness tau2(2, wta_kt_axes, normalized_measure);
    fastjet::contrib::Nsubjettiness tau3(3, wta_kt_axes, normalized_measure);

    result->tau_1 = tau1.result(trimmed_jet);
    result->tau_2 = tau2.result(trimmed_jet);
    result->tau_3 = tau3.result(trimmed_jet);
  }
  return result;
}
