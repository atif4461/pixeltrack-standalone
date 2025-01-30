#include "CUDACore/cudaCheck.h"
#include "HelixFitOnGPU.h"

/**
 * Allocates memory on GPU for helix fit calculation
 * @param[in] tuples input data in form of tuples
 * @param[in] tupleMultiplicity multiplicity of each tuple
 * @param[out] helix_fit_results container for storing results of helix fit
 */
// The above comment was written by an LLM. 
void HelixFitOnGPU::allocateOnGPU(Tuples const *tuples,
                                  TupleMultiplicity const *tupleMultiplicity,
                                  OutputSoA *helix_fit_results) {
  tuples_ = tuples;
  tupleMultiplicity_ = tupleMultiplicity;
  outputSoa_ = helix_fit_results;

  assert(tuples_);
  assert(tupleMultiplicity_);
  assert(outputSoa_);
}

void HelixFitOnGPU::deallocateOnGPU() {}
